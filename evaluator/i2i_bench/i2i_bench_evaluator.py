"""
I2I-Bench Single-Reference Evaluation Dimensions Module

This module implements the 15 single-reference evaluation dimensions from the I2I-Bench paper:
"I2I-Bench: A Comprehensive Benchmark Suite for Image-to-Image Editing Models"

The 15 dimensions are split into:
- 7 Common Dimensions (applicable to all SE tasks)
- 8 Specific Dimensions (for specific instruction types)

Evaluation Methods Used:
- Specialist Models: Q-Insight, ArtiMuse
- LMM VQA-5Level: Qwen3-VL-8B-Instruct
- LMM VQA-2Level: Qwen3-VL-8B-Instruct
- LMM Multi-Question VQA: Qwen3-VL-8B-Instruct
- Hybrid LMM-Specialist: Sa2VA + DINOv3
- Hybrid OCR: GOT-OCR2.0 + LMM
"""

import os
import sys
import json
import base64
import re
import math
from io import BytesIO
from typing import Dict, Any, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from PIL import Image
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ==============================================================================
# COMPREHENSIVE COMPATIBILITY PATCHES for transformers 5.0+
# These must be applied BEFORE importing transformers modules that use older APIs
# ==============================================================================

def _apply_transformers_compatibility_patches():
    """
    Apply monkey patches to transformers for backward compatibility with older model code.
    This is needed for models like q-future/one-align that use deprecated APIs.
    Must be called BEFORE loading such models.
    """
    import torch as _torch
    
    # 1. Add find_pruneable_heads_and_indices to pytorch_utils
    def find_pruneable_heads_and_indices(heads, n_heads, head_dim, already_pruned_heads):
        """Compatibility shim for older model code."""
        mask = _torch.ones(n_heads, head_dim)
        heads = set(heads) - already_pruned_heads
        for head in heads:
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = _torch.arange(len(mask))[mask].long()
        return heads, index
    
    def prune_linear_layer(layer, index, dim=0):
        """Compatibility shim for older model code."""
        index = index.to(layer.weight.device)
        W = layer.weight.index_select(dim, index).clone().detach()
        if layer.bias is not None:
            if dim == 1:
                b = layer.bias.clone().detach()
            else:
                b = layer.bias[index].clone().detach()
        new_size = list(layer.weight.size())
        new_size[dim] = len(index)
        new_layer = _torch.nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
        new_layer.weight.requires_grad = False
        new_layer.weight.copy_(W.contiguous())
        new_layer.weight.requires_grad = True
        if layer.bias is not None:
            new_layer.bias.requires_grad = False
            new_layer.bias.copy_(b.contiguous())
            new_layer.bias.requires_grad = True
        return new_layer
    
    # Add to pytorch_utils module
    try:
        import transformers.pytorch_utils as pytorch_utils
        if not hasattr(pytorch_utils, 'find_pruneable_heads_and_indices'):
            pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
        if not hasattr(pytorch_utils, 'prune_linear_layer'):
            pytorch_utils.prune_linear_layer = prune_linear_layer
    except ImportError:
        pass
    
    # 2. Add Cache to llama module namespace for star imports
    try:
        from transformers.cache_utils import Cache
        import transformers.models.llama.modeling_llama as llama_module
        if not hasattr(llama_module, 'Cache'):
            llama_module.Cache = Cache
    except ImportError:
        pass
    
    # 3. Add get_head_mask back to PreTrainedModel (removed in transformers 5.x)
    #    Self-contained implementation (no dependency on _convert_head_mask_to_5d)
    try:
        from transformers import PreTrainedModel
        if not hasattr(PreTrainedModel, 'get_head_mask'):
            def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
                if head_mask is not None:
                    import torch as _t
                    if head_mask.dim() == 1:
                        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
                    elif head_mask.dim() == 2:
                        head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                    if is_attention_chunked:
                        head_mask = head_mask.unsqueeze(-1)
                else:
                    head_mask = [None] * num_hidden_layers
                return head_mask
            PreTrainedModel.get_head_mask = _get_head_mask
    except ImportError:
        pass

    # 4. Add internal Llama classes to __all__ for star imports
    try:
        import transformers.models.llama.modeling_llama as llama_module
        # These are used by q-future/one-align but not exported via __all__
        internal_classes = ['LlamaMLP', 'LlamaRMSNorm', 'LlamaDecoderLayer', 'LlamaAttention']
        for cls_name in internal_classes:
            if hasattr(llama_module, cls_name) and cls_name not in getattr(llama_module, '__all__', []):
                # Add to module's namespace explicitly
                pass  # Already available, just not in __all__
    except ImportError:
        pass

    # 5. Patch PreTrainedModel._init_weights to inject _use_flash_attention_2 and _use_sdpa
    #    These attributes were set by __init__ in older transformers but removed in 5.x.
    #    Older trust_remote_code models check these attributes directly.
    try:
        from transformers import PreTrainedModel
        _orig_init = PreTrainedModel.__init__

        def _patched_init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            if not hasattr(self, '_use_flash_attention_2'):
                self._use_flash_attention_2 = False
            if not hasattr(self, '_use_sdpa'):
                self._use_sdpa = False
            # all_tied_weights_keys is new in transformers 5.x, older trust_remote_code
            # models don't define it. Provide empty default.
            if not hasattr(self, 'all_tied_weights_keys'):
                self.all_tied_weights_keys = {}

        PreTrainedModel.__init__ = _patched_init
    except Exception:
        pass

# Apply patches immediately
if TORCH_AVAILABLE:
    _apply_transformers_compatibility_patches()

try:
    from transformers import (
        AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoProcessor,
        Qwen2VLForConditionalGeneration, Qwen3VLForConditionalGeneration
    )
    from qwen_vl_utils import process_vision_info
    TRANSFORMERS_AVAILABLE = True
        
except ImportError as e:
    print(f"Warning: Could not import transformers: {e}")
    TRANSFORMERS_AVAILABLE = False


# ==============================================================================
# CONFIGURATION AND CONSTANTS
# ==============================================================================

@dataclass
class EvaluatorConfig:
    """Configuration for the I2I-Bench evaluator - EXACT models from the paper."""
    # LMM Model (Generalist) - EXACT from paper Section A.2
    lmm_model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    
    # Specialist Models - EXACT from paper Section A.2
    q_insight_model_name: str = "q-future/one-align"  # OneAlign for image quality scoring
    artimuse_model_name: str = "Thunderbolt215215/ArtiMuse"  # ArtiMuse for aesthetics
    got_ocr_model_name: str = "stepfun-ai/GOT-OCR-2.0-hf"  # GOT-OCR2.0 HuggingFace version
    sa2va_model_name: str = "ByteDance/Sa2VA-4B"  # Sa2VA for segmentation
    dinov3_model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"  # DINOv3 (NOT DINOv2)
    
    # Device
    device: str = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
    
    # VQA Score mappings
    vqa_5level_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.vqa_5level_weights is None:
            self.vqa_5level_weights = {
                "excellent": 1.0,
                "good": 0.75,
                "fair": 0.5,
                "poor": 0.25,
                "bad": 0.0
            }


# ==============================================================================
# PROMPTS - Exactly as specified in I2I-Bench paper (Appendix F.1)
# ==============================================================================

class Prompts:
    """All prompts from the I2I-Bench paper for single-reference evaluation."""
    
    # =========================================================================
    # COMMON DIMENSIONS (7 dimensions)
    # =========================================================================
    
    # 3. Blending Naturalness - LMM VQA-5Level
    BLENDING_NATURALNESS_SYSTEM = """You are an expert evaluator of image photorealism and coherence. Your specific task is to assess the realism of the edit itself. Evaluate how seamlessly the modified or added elements integrate with the rest of the image in terms of lighting, shadows, perspective, and texture. A high rating means the final image looks natural and plausible, as if it were a single, untouched photograph. Do not focus on whether the instruction was followed literally. Your response must be one of the following five words directly: excellent, good, fair, poor, bad. Do not add any introductory phrases."""
    
    BLENDING_NATURALNESS_USER = """Please evaluate the realism and coherence of the edit in the 'Generated Image' compared to the 'Source Images'. Assess how seamlessly the edited elements integrate with the rest of the image in terms of lighting, shadows, and overall plausibility. Your response must begin with one of the five rating words: excellent, good, fair, poor, bad."""
    
    # 4. Generative/Editing Artifacts - LMM VQA-5Level
    EDITING_ARTIFACTS_SYSTEM = """You are an expert evaluator of image editing quality. You will be given an 'Original Image', an 'Editing Instruction', and the resulting 'Edited Image'. Your task is to assess the 'Edited Image' for *unwanted artifacts* introduced *during the editing process*, not artifacts that were in the original.

**Crucially**: If the instruction is stylistic (e.g., 'turn into a painting', 'make it look like Van Gogh'), *do not* penalize the image for looking 'unnatural'. Instead, judge if the *application* of the style is flawed (e.g., distorted, incomplete, blotchy).

Also, heavily penalize edits that *fail to preserve* unedited regions (e.g., if the instruction is 'change the woman's hat', her face and the background should remain unchanged).

Based on the *severity* of these *unwanted* artifacts, rate the 'Edited Image' using *only* one of the following five words:
1. excellent (Flawless edit. The instruction is followed perfectly with *zero* unwanted artifacts. Unedited areas are perfectly preserved.)
2. good (Minor, hard-to-notice artifacts. The edit is successful but may have tiny imperfections upon close inspection.)
3. fair (Noticeable artifacts. The edit is mostly successful, but there are visible flaws like slight warping, minor texture loss, or imperfect blending.)
4. poor (Significant, obvious artifacts. The edit is flawed, with clear distortions, unnatural warping, or significant damage to unedited areas.)
5. bad (Extreme, severe artifacts. The edit is a total failure, resulting in a grotesque, distorted, or nonsensical image.)"""
    
    EDITING_ARTIFACTS_USER = """Based on the instruction, how severe are the *unwanted artifacts* in the 'Edited Image'?"""
    
    # 5. Instruction Following (Macro) - LMM VQA-5Level
    INSTRUCTION_FOLLOWING_SYSTEM = """You are a meticulous evaluator specializing in text-to-image editing. Your sole task is to assess how accurately the edited image reflects the given instruction, based on the original image. Focus exclusively on whether the edit described in the instruction was performed correctly. You must ignore all other factors, such as overall image quality or any unintended changes in areas not mentioned in the instruction. Your response must be one of the following five words directly: excellent, good, fair, poor, bad. Do not add any introductory phrases."""
    
    INSTRUCTION_FOLLOWING_USER = """Please evaluate whether the 'Generated Image' successfully implements the following instruction. Instruction: "{prompt}". Your response must begin with one of the five rating words: excellent, good, fair, poor, bad."""
    
    # 6. Non-Edited Region Fidelity - Hybrid LMM-Specialist
    NON_EDITED_REGION_SEGMENTATION = """You are an image segmentation assistant. Compare <image_1> (Original Image) and <image_2> (Edited Image), and considering the editing instruction: "{edit_instruction}", generate a text command to segment the edited or modified region in the image. Your response MUST begin with "Please segment". This command will be used to segment the edited image in isolation. Therefore, ensure the command is clear, focuses only on the edited image, and makes no reference to the original image, as this would confuse the segmentation model."""
    
    # 7. Physical Plausibility - LMM Multi-Question VQA
    PHYSICAL_PLAUSIBILITY_QG = """You are an expert in physical plausibility evaluation. Your task is to design a set of (5 to 7) precise, specific binary (Yes/No) questions based on the given original image and an editing instruction, to evaluate whether the edited image is physically plausible.

When designing questions, strictly follow this thought process and principles:

1. **Analyze Scene & Instruction**:
   * **Identify Subject & Action**: What object (subject) is being manipulated (added, removed, replaced, attribute changed)?
   * **Analyze Physical Context**: Observe the original image to identify relevant physical properties and environmental factors: e.g., lighting direction, shadows, reflective surfaces, support relationships, rigid vs. soft bodies.

2. **Generate Questions**:
   Based on the analysis, generate questions from the following physical dimensions. Each must be a closed-ended Yes/No question.
   * **Optics (Shadows, Reflections)**: If an object was added/moved, is its new shadow consistent with the scene's light source? If an object was removed, is its shadow also gone? Are reflections on nearby surfaces correctly updated?
   * **Mechanics (Deformation, Support)**: If a heavy object is placed on a soft surface, does the surface show plausible deformation? If a supporting object is removed, does the object it supported (e.g., a vase on a table) defy gravity?
   * **State Transition (Weather, State)**: If the instruction changes the weather (e.g., 'make it winter'), are all elements (trees, ground) consistently updated (e.g., covered in snow)?

3. **Output Format**: Please output a JSON-formatted string containing a list of questions. The JSON object should have a key named "physical_questions" with a value that is a list of strings.
Example: {{ "physical_questions": [ "Question 1: ...?", "Question 2: ...?", "Question 3: ...?" ] }}

Now, please generate physical plausibility evaluation questions based on the following image and instruction. Instruction: "{prompt}"."""

    PHYSICAL_PLAUSIBILITY_QA = """You are an expert in physical plausibility evaluation. Please carefully observe the provided image and judge based on the following question.

Question: "{question}"

Please answer with only one word based on visual evidence: Yes or No."""

    # =========================================================================
    # SPECIFIC DIMENSIONS (8 dimensions)
    # =========================================================================
    
    # 8-11. Object Manipulation, Local Attribute, Action/State Change, Spatial Accuracy - LMM VQA-2Level
    VQA_2LEVEL_SYSTEM = """You are an expert in evaluating image editing. Your task is to determine if the edited image successfully implements the given instruction compared to the original image. Focus solely on whether the instruction was achieved. Answer strictly and only with 'Yes' or 'No'."""
    
    VQA_2LEVEL_USER = """Instruction: "{prompt}". Does Image 2 successfully implement the instruction compared to Image 1? Answer Yes or No."""
    
    # 12. Text Content & Style Accuracy - Hybrid OCR + LMM
    TEXT_STYLE_PROMPT = """You are an expert evaluator for AI-generated images, specializing in text rendering. Your task is to evaluate how well an image follows the text-related **style and position** instructions from a user prompt.

**CRITICAL RULE:** Do NOT evaluate the text's spelling or accuracy. Assume the spelling is correct, even if it is not. Your score must ONLY reflect the non-accuracy requirements (like font style, color, placement, etc.).

You will be given:
1. **[User Prompt]**: The original prompt used to generate the image.
2. **[Target Text]**: The specific text string that was requested.
3. **[Image]**: The generated image.

**Step 1: Analyze Requirements**
First, analyze the [User Prompt] to identify the specific requirements for:
* **Text Style**: What instructions were given for the text's appearance (e.g., "neon", "handwritten", "bold", "red color", "glowing", "artistic font")?
* **Text Position**: What instructions were given for the text's location (e.g., "on the sign", "in the top-left corner", "on the t-shirt")?

**Step 2: Evaluate Image against Requirements**
Compare the text in the [Image] against the non-accuracy requirements you identified. Remember to IGNORE spelling errors.

**Step 3: Assign a Single Score (1-5)**
Provide a single, holistic score for **Style and Position Compliance** based on this rubric.
* [5] Excellent Match: All specified style and position instructions were followed perfectly.
* [4] Good Match: All specified instructions were followed, but with minor deviations.
* [3] Partial Match: The core idea of *at least one* instruction was attempted but executed poorly, OR one major instruction was followed while another was missed.
* [2] Poor Match: At least one specified instruction was clearly ignored or failed.
* [1] No Match: All specified style and position instructions were completely ignored.

**Step 4: Provide Output in JSON Format**
Provide your evaluation in a strict JSON format. Do not include any text outside the JSON block.

**JSON Output Format:**
{{ "analysis": {{ "style_requirement": "...", "position_requirement": "...", "image_observation": "...", "reasoning": "...", "score": [1-5] }} }}"""
    
    # 13. World Knowledge & Reasoning - LMM VQA-2Level
    WORLD_KNOWLEDGE_SYSTEM = """You are an expert evaluator for image editing based on multiple reference images. Your task is to determine if the 'Edited Image' successfully implements the requested change described in the 'Instruction', based *specifically* on the 'Evaluation Criteria (Hint)'. Compare the 'Reference Images' and 'Edited Image'. Your response must be *only* the single word 'Yes' or 'No'.

'Yes' = The edit was successfully implemented according to the hint.
'No' = The edit was not successfully implemented according to the hint."""
    
    WORLD_KNOWLEDGE_USER = """**Instruction (Prompt):** {prompt}

**Evaluation Criteria (Hint):** {hint}

Based on ALL reference images, the instruction, and the specific criteria in the hint, has the edit been successfully implemented in the 'Edited Image'? Answer with only 'Yes' or 'No'."""
    
    # 14. Subject Identity Fidelity - Hybrid LMM-Specialist
    SUBJECT_IDENTITY_SEGMENTATION = """You are an expert in image editing analysis. Given an original image, an edited image, and the editing instruction, identify all distinct main subjects or regions present in the original image that *should remain unchanged* according to the instruction.

Your output MUST BE ONLY a JSON list of strings. Each string must be a separate segmentation command for one distinct subject/region, in the format 'Please segment [subject name in English]'.

For example: ["Please segment background sky", "Please segment mountains", "Please segment main building"]"""
    
    # 15. Composition & Interaction - LMM Multi-Question VQA
    COMPOSITION_INTERACTION_QG = """You are an expert in visual arts and composition evaluation. Your task is to design a set of (5 to 7) precise, specific binary (Yes/No) questions based on the given original image and an editing instruction, to evaluate whether the edited image is plausible in terms of **composition, perspective, scale, and interaction**.

When designing questions, strictly follow this thought process and principles:

1. **Analyze Scene & Instruction**:
   * **Identify Subject & Action**: What object (subject) is being manipulated (added, removed, replaced, attribute/pose changed)?
   * **Analyze Visual Context**: Observe the original image to identify key visual elements: scene perspective (close-up, long-shot, eye-level), key objects, spatial layout (foreground/background), and interaction area implied by the instruction.

2. **Generate Questions**:
   Based on the analysis, generate questions from the following dimensions.
   * **Composition & Placement**: Is the new/moved object in a logical position? Is its occlusion (in front of/behind other objects) correct?
   * **Perspective & Scale**: Is the scale of the new/modified object consistent with other objects in the scene? Does its perspective match the scene's perspective?
   * **Interaction & Naturalness**: If a pose was changed, is it anatomically natural? If objects are interacting (e.g., hand holding a balloon), is the contact point believable?

3. **Output Format**: Please output a JSON-formatted string containing a list of questions. The JSON object should have a key named "composition_questions" with a value that is a list of strings.

Now, please generate composition and interaction plausibility evaluation questions based on the following image and instruction. Instruction: "{prompt}"."""


def clear_gpu_memory():
    """Clear GPU memory and reset all singleton handlers."""
    import gc
    import sys
    import ctypes
    
    current_module = sys.modules[__name__]
    
    handler_names = [
        'LMMHandler', 'QInsightHandler', 'ArtiMuseHandler', 
        'GOTOCRHandler', 'Sa2VAHandler', 'DINOv3Handler'
    ]
    
    for handler_name in handler_names:
        try:
            handler_class = getattr(current_module, handler_name, None)
            if handler_class is None:
                continue
            
            instance = getattr(handler_class, '_instance', None)
            if instance is None:
                continue
            
            # Delete model/tokenizer/processor attributes aggressively
            for attr in ['_model', '_tokenizer', '_processor', '_image_processor']:
                model_ref = getattr(instance, attr, None)
                if model_ref is not None:
                    try:
                        setattr(instance, attr, None)
                    except:
                        pass
                    del model_ref
            
            # Also clear class-level attributes
            for attr in ['_model', '_tokenizer', '_processor', '_image_processor']:
                class_ref = getattr(handler_class, attr, None)
                if class_ref is not None:
                    try:
                        setattr(handler_class, attr, None)
                    except:
                        pass
            
            instance._initialized = False
            handler_class._instance = None
            del instance
                    
        except Exception as e:
            print(f"Warning: Error clearing {handler_name}: {e}")
    
    # Aggressive garbage collection
    gc.collect()
    gc.collect()
    gc.collect()
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



# ==============================================================================
# BASE CLASSES
# ==============================================================================

class BaseEvaluator(ABC):
    """Abstract base class for all evaluation dimensions."""
    
    def __init__(self, config: EvaluatorConfig):
        self.config = config
        self.models = {}
    
    @abstractmethod
    def evaluate(self, original_image: Image.Image, edited_image: Image.Image, 
                 instruction: str, **kwargs) -> Dict[str, Any]:
        """
        Evaluate an image editing result.
        
        Args:
            original_image: The original input image
            edited_image: The edited/generated output image
            instruction: The editing instruction/prompt
            **kwargs: Additional arguments specific to each dimension
            
        Returns:
            Dictionary containing 'score' and optional 'details'
        """
        pass
    
    @property
    @abstractmethod
    def dimension_name(self) -> str:
        """Return the name of this evaluation dimension."""
        pass
    
    def load_image(self, image_path: str) -> Image.Image:
        """Load an image from path."""
        return Image.open(image_path).convert("RGB")
    
    def image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()


# ==============================================================================
# LMM (Large Multimodal Model) HANDLER - Qwen3-VL-8B-Instruct (EXACT from paper)
# ==============================================================================

class LMMHandler:
    """
    Handler for the Generalist LMM: Qwen3-VL-8B-Instruct (EXACT from I2I-Bench paper).
    Used for VQA-5Level, VQA-2Level, and Multi-Question VQA pipelines.
    NO FALLBACKS - raises error if model cannot be loaded.
    """
    
    _instance = None
    _model = None
    _processor = None
    
    def __new__(cls, config: EvaluatorConfig):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: EvaluatorConfig):
        self.config = config
        self._initialized = False
    
    def initialize(self):
        """Lazy initialization of the model. Raises error on failure."""
        if self._initialized:
            return
        
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers library is required for LMM. Install with: pip install transformers")
        
        print(f"Loading LMM: {self.config.lmm_model_name}")
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.config.lmm_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self._processor = AutoProcessor.from_pretrained(
            self.config.lmm_model_name,
            trust_remote_code=True
        )
        self._initialized = True
        print("LMM (Qwen3-VL-8B-Instruct) loaded successfully.")
    
    def generate(self, images: List[Image.Image], system_prompt: str, 
                 user_prompt: str, max_tokens: int = 512) -> str:
        """
        Generate response from the LMM.
        
        Args:
            images: List of PIL Images to include in the prompt
            system_prompt: System prompt text
            user_prompt: User prompt text
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        self.initialize()
        
        # Build messages for Qwen3-VL
        content = []
        for i, img in enumerate(images):
            # Convert image to base64
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            content.append({
                "type": "image",
                "image": f"data:image/png;base64,{img_base64}"
            })
        content.append({"type": "text", "text": user_prompt})
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]
        
        # Process and generate
        # enable_thinking=False prevents Qwen3-VL from outputting <think>...</think> tags
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self._model.device)
        
        generated_ids = self._model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return output_text.strip()
    
    def get_logits_for_choices(self, images: List[Image.Image], system_prompt: str,
                               user_prompt: str, choices: List[str]) -> Dict[str, float]:
        """
        Get the probability distribution over a set of choices.
        Used for VQA-5Level and VQA-2Level scoring.
        
        Returns:
            Dictionary mapping each choice to its probability
        """
        self.initialize()
        
        # Generate response and extract the choice
        response = self.generate(images, system_prompt, user_prompt, max_tokens=10)
        response_lower = response.lower().strip()
        
        # Find which choice matches
        probs = {}
        matched = False
        for choice in choices:
            if choice.lower() in response_lower or response_lower.startswith(choice.lower()):
                probs[choice] = 1.0
                matched = True
            else:
                probs[choice] = 0.0
        
        # If no exact match, distribute based on response similarity
        if not matched:
            # Default to middle choice
            mid_idx = len(choices) // 2
            for i, choice in enumerate(choices):
                probs[choice] = 1.0 if i == mid_idx else 0.0
        
        return probs


# ==============================================================================
# SPECIALIST MODELS (EXACT from I2I-Bench paper - NO FALLBACKS)
# ==============================================================================

def _patch_q_insight_model_files():
    """
    Comprehensive patch for q-future/one-align cached model files.
    Fixes transformers 5.x compatibility issues:
    1. Explicit imports (star import filtered by __all__)
    2. Logger definition
    3. Backward-compatible LlamaRotaryEmbedding (old dim/max_pos_emb/base API)
    4. Backward-compatible apply_rotary_pos_emb (old position_ids API)
    5. rope_scaling config key format changes
    6. Config class-level defaults (mlp_bias, attention_dropout)
    """
    import glob
    import os
    import shutil

    cache_dirs = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/modules/transformers_modules/q_hyphen_future/one_hyphen_align/*"
    ))
    # Also try non-encoded path format
    cache_dirs += glob.glob(os.path.expanduser(
        "~/.cache/huggingface/modules/transformers_modules/q-future/one-align/*"
    ))
    
    if not cache_dirs:
        print("Warning: No q-future/one-align cache directories found to patch")

    for cache_dir in cache_dirs:
        if not os.path.isdir(cache_dir):
            continue

        # === Patch modeling_llama2.py ===
        llama_file = os.path.join(cache_dir, "modeling_llama2.py")
        if os.path.exists(llama_file):
            try:
                with open(llama_file, 'r') as f:
                    content = f.read()

                modified = False
                
                # Apply star-import compat patch if not yet done
                if "# === I2IBench transformers 5.x compat patch ===" not in content:
                    old_star = "from transformers.models.llama.modeling_llama import *"
                    compat_imports = '''from transformers.models.llama.modeling_llama import *
# === I2IBench transformers 5.x compat patch ===
from transformers.models.llama.modeling_llama import (
    LlamaAttention as _LlamaAttention5x,
    LlamaDecoderLayer as _LlamaDecoderLayer5x,
    LlamaForCausalLM, LlamaMLP, LlamaModel, LlamaRMSNorm,
    repeat_kv, rotate_half,
)
try:
    from transformers.cache_utils import Cache
except ImportError:
    from typing import Any as Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

# Backward-compatible LlamaRotaryEmbedding (transformers 4.x API)
class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device,
                                dtype=torch.get_default_dtype())
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (self.cos_cached[:seq_len].to(dtype=x.dtype), self.sin_cached[:seq_len].to(dtype=x.dtype))

class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype) / self.scaling_factor
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        if seq_len > self.max_position_embeddings:
            base = self.base * ((self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

# Backward-compatible apply_rotary_pos_emb (transformers 4.x API with position_ids)
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
'''
                    content = content.replace(old_star, compat_imports, 1)
                    # Remove any second star import
                    if old_star in content:
                        content = content.replace(old_star, "# (duplicate star import removed)", 1)
                    modified = True

                # Always apply logger fix (idempotent check)
                if "logger = logging.get_logger(__name__)" not in content:
                    content = content.replace(
                        "from transformers.utils import logging",
                        "from transformers.utils import logging\nlogger = logging.get_logger(__name__)",
                        1
                    )
                    modified = True

                # Always fix rope_scaling config key access for transformers 5.x
                if 'self.config.rope_scaling["type"]' in content:
                    content = content.replace(
                        'if self.config.rope_scaling is None:',
                        'if self.config.rope_scaling is None or self.config.rope_scaling.get("rope_type", self.config.rope_scaling.get("type")) == "default":'
                    )
                    content = content.replace(
                        'scaling_type = self.config.rope_scaling["type"]',
                        'scaling_type = self.config.rope_scaling.get("type", self.config.rope_scaling.get("rope_type", "default"))'
                    )
                    content = content.replace(
                        'scaling_factor = self.config.rope_scaling["factor"]',
                        'scaling_factor = self.config.rope_scaling.get("factor", 1.0)'
                    )
                    modified = True

                if modified:
                    with open(llama_file, 'w') as f:
                        f.write(content)
                    print(f"Patched {llama_file} for transformers 5.x compatibility")
            except Exception as e:
                print(f"Warning: Could not patch {llama_file}: {e}")

        # === Patch configuration_mplug_owl2.py ===
        config_file = os.path.join(cache_dir, "configuration_mplug_owl2.py")
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    content = f.read()

                config_modified = False

                # Add logger if not present
                if "logger = logging.get_logger(__name__)" not in content and "logger = _i2i_logging.getLogger" not in content:
                    content = content.replace(
                        "from transformers.utils import logging",
                        "from transformers.utils import logging\nlogger = logging.get_logger(__name__)",
                        1
                    )
                    config_modified = True

                # Add class-level defaults for missing transformers 5.x config attributes
                if "# === I2IBench config compat ===" not in content:
                    content = content.replace(
                        "class LlamaConfig(PretrainedConfig):",
                        "class LlamaConfig(PretrainedConfig):\n"
                        "    # === I2IBench config compat ===\n"
                        "    mlp_bias = False\n"
                        "    attention_dropout = 0.0",
                        1
                    )
                    config_modified = True

                if config_modified:
                    with open(config_file, 'w') as f:
                        f.write(content)
                    print(f"Patched {config_file} for transformers 5.x compatibility")
            except Exception as e:
                print(f"Warning: Could not patch {config_file}: {e}")

        # === Ensure ALL .py files that use 'logger' have it defined ===
        for py_file in glob.glob(os.path.join(cache_dir, "*.py")):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                # Check if file uses 'logger' but doesn't define it
                if 'logger' in content and 'logger = logging.get_logger' not in content and 'logger =' not in content:
                    # Add logger definition at the top after imports
                    logger_code = "\nimport logging\nlogger = logging.getLogger(__name__)\n"
                    # Insert after the last import line
                    lines = content.split('\n')
                    last_import_idx = 0
                    for i, line in enumerate(lines):
                        if line.startswith('import ') or line.startswith('from '):
                            last_import_idx = i
                    lines.insert(last_import_idx + 1, logger_code)
                    content = '\n'.join(lines)
                    with open(py_file, 'w') as f:
                        f.write(content)
                    print(f"  Injected logger into {os.path.basename(py_file)}")
            except Exception as e:
                print(f"  Warning: Could not check logger in {py_file}: {e}")

        # Clear __pycache__
        pc = os.path.join(cache_dir, "__pycache__")
        if os.path.exists(pc):
            shutil.rmtree(pc)

        # === Universal logger injection: scan ALL .py files for 'logger' usage ===
        for py_file in glob.glob(os.path.join(cache_dir, "*.py")):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                # If file uses 'logger' but doesn't define it, inject at top
                if 'logger' in content and 'logger = ' not in content and 'logger=' not in content:
                    # Add logger definition after imports
                    inject = "import logging as _i2i_logging\nlogger = _i2i_logging.getLogger(__name__)\n"
                    if inject not in content:
                        content = inject + content
                        with open(py_file, 'w') as f:
                            f.write(content)
                        print(f"  Injected logger into {os.path.basename(py_file)}")
            except Exception as e:
                print(f"  Warning: Could not inject logger into {py_file}: {e}")
        
        # Clear __pycache__ again after logger injection
        if os.path.exists(pc):
            shutil.rmtree(pc)


def _patch_internvl_meta_tensor_issue():
    """
    Patch InternVL-based models (ArtiMuse, Sa2VA) for transformers 5.x compatibility.
    
    In transformers 5.x, from_pretrained creates models on meta device first.
    InternVL's modeling_intern_vit.py and sam2.py use torch.linspace().item()
    in __init__ which fails on meta tensors. Replace with numpy.linspace.
    """
    import glob
    import shutil
    
    old_pattern = "[x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]"
    new_pattern = "__import__('numpy').linspace(0, float(config.drop_path_rate) if not isinstance(config.drop_path_rate, (int, float)) else config.drop_path_rate, config.num_hidden_layers).tolist()"
    
    sam2_old = "x.item() for x in torch.linspace(0, drop_path_rate, depth)"
    sam2_new = "float(x) for x in __import__('numpy').linspace(0, float(drop_path_rate) if not isinstance(drop_path_rate, (int, float)) else drop_path_rate, depth)"
    
    # Find all relevant cached model files (try both encoded and non-encoded paths)
    patterns = [
        os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/Thunderbolt215215/ArtiMuse/*/modeling_intern_vit.py"),
        os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/ByteDance/Sa2VA_hyphen_4B/*/modeling_intern_vit.py"),
        os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/ByteDance/Sa2VA_hyphen_4B/*/sam2.py"),
        os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/ByteDance/Sa2VA-4B/*/modeling_intern_vit.py"),
        os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/ByteDance/Sa2VA-4B/*/sam2.py"),
    ]
    
    for pattern in patterns:
        for fpath in glob.glob(pattern):
            try:
                with open(fpath, 'r') as f:
                    content = f.read()
                
                modified = False
                if old_pattern in content:
                    content = content.replace(old_pattern, new_pattern)
                    modified = True
                if sam2_old in content:
                    content = content.replace(sam2_old, sam2_new)
                    modified = True
                
                if modified:
                    with open(fpath, 'w') as f:
                        f.write(content)
                    # Clear __pycache__
                    pc = os.path.join(os.path.dirname(fpath), "__pycache__")
                    if os.path.exists(pc):
                        shutil.rmtree(pc)
                    print(f"Patched meta tensor issue in {fpath}")
            except Exception as e:
                print(f"Warning: Could not patch {fpath}: {e}")


def _patch_artimuse_score_method():
    """
    Patch ArtiMuse cached model to add score() and generate_logits() methods.
    
    The HuggingFace cached version of ArtiMuse (modeling_internvl_chat.py) lacks
    the score() method that exists in the GitHub source (modeling_artimuse.py).
    This function:
    1. Creates aes_tokens.py with AESTHETICS_TOKEN_LIST (101 tokens mapping 0-100)
    2. Adds 'from .aes_tokens import AESTHETICS_TOKEN_LIST' import
    3. Adds generate_logits() method (forward pass instead of generate())
    4. Adds score() method (aesthetic prompt -> forward pass -> softmax -> weighted avg)
    """
    import glob
    import shutil
    
    cache_dirs = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/modules/transformers_modules/Thunderbolt215215/ArtiMuse/*"
    ))
    
    if not cache_dirs:
        print("Warning: No ArtiMuse cache directories found for score() patching")
    
    for cache_dir in cache_dirs:
        if not os.path.isdir(cache_dir):
            continue
        
        # 1. Create aes_tokens.py if not present
        aes_tokens_path = os.path.join(cache_dir, "aes_tokens.py")
        if not os.path.exists(aes_tokens_path):
            aes_tokens_content = '''def score2aestoken(n):
    if not (0 <= n <= 100):
        raise ValueError("Score must be between 0 and 100 inclusive.")
    
    if 0 <= n <= 25:
        first = 'a'
        offset = n
    elif 26 <= n <= 50:
        first = 'c'
        offset = n - 26
    elif 51 <= n <= 75:
        first = 'd'
        offset = n - 51
    else:
        first = 'e'
        offset = n - 76

    second = chr(ord('a') + offset)
    return first + second

AESTHETICS_TOKEN_LIST = []

for i in range(101):
    AESTHETICS_TOKEN_LIST.append(score2aestoken(i))
'''
            with open(aes_tokens_path, 'w') as f:
                f.write(aes_tokens_content)
            print(f"Created {aes_tokens_path}")
        
        # 2. Patch modeling_internvl_chat.py
        model_file = os.path.join(cache_dir, "modeling_internvl_chat.py")
        if not os.path.exists(model_file):
            continue
        
        with open(model_file, 'r') as f:
            content = f.read()
        
        modified = False
        
        # Add import if not present
        if "from .aes_tokens import AESTHETICS_TOKEN_LIST" not in content:
            # Insert after the existing imports
            anchor = "from .modeling_intern_vit import InternVisionModel"
            if anchor in content:
                # Find the line containing the anchor and insert after it
                lines = content.split('\n')
                new_lines = []
                for line in lines:
                    new_lines.append(line)
                    if anchor in line:
                        # Skip to after the line, then add import
                        pass
                # Simpler: just insert after the anchor line
                content = content.replace(
                    anchor,
                    anchor + "\n\nfrom .aes_tokens import AESTHETICS_TOKEN_LIST"
                )
                modified = True
        
        # Add generate_logits() and score() if not present
        if "def generate_logits(" not in content:
            # Find the end of the generate() method to insert after it
            # Look for the return statement pattern at the end of generate()
            generate_end_marker = "        return outputs\n"
            # Find the LAST occurrence (generate method's return)
            idx = content.rfind(generate_end_marker)
            if idx == -1:
                # Try alternate pattern
                generate_end_marker = "        return outputs"
                idx = content.rfind(generate_end_marker)
            
            if idx != -1:
                insert_pos = idx + len(generate_end_marker)
                # If there's a newline after, keep it
                if insert_pos < len(content) and content[insert_pos] == '\n':
                    insert_pos += 1
                
                new_methods = '''
    @torch.no_grad()
    def generate_logits(
            self,
            pixel_values=None,
            input_ids=None,
            attention_mask=None,
            visual_features=None,
            generation_config=None,
            output_hidden_states=None,
            **generate_kwargs,
    ):
        """Get logits by doing a forward pass instead of generate, for scoring."""
        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        # Use forward pass to get logits instead of generate()
        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
        )

        return outputs

    def score(self,
            device,
            tokenizer,
            pixel_values,
            generation_config,
            history=None,
            num_patches_list=None,
            IMG_START_TOKEN=\'<img>\',
            IMG_END_TOKEN=\'</img>\',
            IMG_CONTEXT_TOKEN=\'<IMG_CONTEXT>\',
            verbose=False):

        question = """
        Rate the aesthetics score of the image in 0-100. 
        In the output format, numbers are replaced by 2 corresponding letters, 
        and the mapping relationship is: 
        score 0 to 25: 0-aa, 1-ab, 2-ac, 3-ad, ... , 25-az,
        score 26 to 50: 26-ca, 27-cb, 28-cc, 29-cd, ..., 50-cy,
        score 51 to 75: 51-da, 52-db, 53-dc, 54-dd, ..., 75-dy,
        score 76 to 100: 76-ea, 77-eb, 78-ec, 79-ed, ..., 100-ey.

        The answer only outputs 2 corresponding letters.
        """

        if history is None and pixel_values is not None and \'<image>\' not in question:
            question = \'<image>\\n\' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id
        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f\'dynamic ViT batch size: {image_bs}\')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace(\'<image>\', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors=\'pt\')

        input_ids = model_inputs[\'input_ids\'].to(device)
        attention_mask = model_inputs[\'attention_mask\'].to(device)
        generation_output = self.generate_logits(pixel_values=pixel_values,
                                                 input_ids=input_ids,
                                                 attention_mask=attention_mask)

        logits = generation_output.logits

        preferential_ids_ = [tokenizer.convert_tokens_to_ids(word) for word in AESTHETICS_TOKEN_LIST]
        output_logits = logits[:, -1, preferential_ids_].detach()
        weight_tensor = torch.tensor([x for x in range(101)]).to(device)

        score = torch.softmax(output_logits, -1) @ weight_tensor.to(output_logits.dtype)
        return score.item()

'''
                content = content[:insert_pos] + new_methods + content[insert_pos:]
                modified = True
        
        if modified:
            with open(model_file, 'w') as f:
                f.write(content)
            # Clear __pycache__
            pc = os.path.join(cache_dir, "__pycache__")
            if os.path.exists(pc):
                shutil.rmtree(pc)
            print(f"Patched ArtiMuse score/generate_logits methods in {model_file}")


def _bind_artimuse_score_methods(model):
    """
    Dynamically bind score() and generate_logits() methods to an ArtiMuse model instance.
    This is the fallback when file-based patching + module cache invalidation doesn't work
    (e.g., transformers caches modules in ways that resist sys.modules cleanup).
    """
    import types
    
    # Build AESTHETICS_TOKEN_LIST
    def score2aestoken(n):
        if 0 <= n <= 25:
            first, offset = 'a', n
        elif 26 <= n <= 50:
            first, offset = 'c', n - 26
        elif 51 <= n <= 75:
            first, offset = 'd', n - 51
        else:
            first, offset = 'e', n - 76
        return first + chr(ord('a') + offset)

    AESTHETICS_TOKEN_LIST = [score2aestoken(i) for i in range(101)]

    @torch.no_grad()
    def generate_logits(self, pixel_values=None, input_ids=None, attention_mask=None,
                         visual_features=None, generation_config=None, output_hidden_states=None,
                         **generate_kwargs):
        assert self.img_context_token_id is not None
        if pixel_values is not None:
            vit_embeds = visual_features if visual_features is not None else self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)
            input_ids_flat = input_ids.reshape(B * N)
            selected = (input_ids_flat == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
        outputs = self.language_model(inputs_embeds=input_embeds, attention_mask=attention_mask)
        return outputs

    def score(self, device, tokenizer, pixel_values, generation_config, history=None,
              num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
              IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False):
        question = """
        Rate the aesthetics score of the image in 0-100. 
        In the output format, numbers are replaced by 2 corresponding letters, 
        and the mapping relationship is: 
        score 0 to 25: 0-aa, 1-ab, 2-ac, 3-ad, ... , 25-az,
        score 26 to 50: 26-ca, 27-cb, 28-cc, 29-cd, ..., 50-cy,
        score 51 to 75: 51-da, 52-db, 53-dc, 54-dd, ..., 75-dy,
        score 76 to 100: 76-ea, 77-eb, 78-ec, 79-ed, ..., 100-ey.

        The answer only outputs 2 corresponding letters.
        """
        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question
        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        # Use get_conv_template from the model's own module
        model_module_name = type(self).__module__
        model_module = sys.modules.get(model_module_name)
        get_conv_template = None
        if model_module and hasattr(model_module, 'get_conv_template'):
            get_conv_template = model_module.get_conv_template
        if get_conv_template is None:
            try:
                from fastchat.conversation import get_conv_template
            except ImportError:
                pass
        if get_conv_template is None:
            # Import from the ArtiMuse conversation.py in the cache
            import importlib
            for key in sys.modules:
                if 'Thunderbolt215215' in key or 'ArtiMuse' in key:
                    mod = sys.modules[key]
                    if hasattr(mod, 'get_conv_template'):
                        get_conv_template = mod.get_conv_template
                        break

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        generation_output = self.generate_logits(pixel_values=pixel_values,
                                                  input_ids=input_ids,
                                                  attention_mask=attention_mask)
        logits = generation_output.logits
        preferential_ids_ = [tokenizer.convert_tokens_to_ids(word) for word in AESTHETICS_TOKEN_LIST]
        output_logits = logits[:, -1, preferential_ids_].detach()
        weight_tensor = torch.tensor([x for x in range(101)]).to(device)
        score_val = torch.softmax(output_logits, -1) @ weight_tensor.to(output_logits.dtype)
        return score_val.item()

    model.generate_logits = types.MethodType(generate_logits, model)
    model.score = types.MethodType(score, model)
    print("  Successfully bound score() and generate_logits() to ArtiMuse model instance")


def _inject_cache_into_llama_module():
    """
    Inject backward-compatible classes into transformers.models.llama.modeling_llama
    before loading q-future/one-align model. The model code uses
    `from transformers.models.llama.modeling_llama import *` which in transformers 5.x
    no longer exports Cache, LlamaRotaryEmbedding, apply_rotary_pos_emb, etc.
    We inject shim versions so the custom code can import them.
    """
    try:
        from transformers.cache_utils import Cache
        from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
        import transformers.models.llama.modeling_llama as llama_module
        
        # --- Backward-compat LlamaRotaryEmbedding (transformers 4.x API) ---
        # Transformers 5.x removed the old LlamaRotaryEmbedding that accepted
        # (dim, max_position_embeddings, base) arguments. The custom modeling_llama2.py
        # needs the old API.
        import torch.nn as nn
        
        # Save the real (transformers 5.x) implementations before overwriting
        _real_LlamaRotaryEmbedding = getattr(llama_module, 'LlamaRotaryEmbedding', None)
        
        class _LlamaRotaryEmbeddingOld(nn.Module):
            """Old-API LlamaRotaryEmbedding: (dim, max_position_embeddings, base, device)"""
            def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
                super().__init__()
                self.dim = dim
                self.max_position_embeddings = max_position_embeddings
                self.base = base
                # Detect meta device context (transformers 5.x creates model on meta)
                _in_meta = (device is not None and str(device) == 'meta')
                if not _in_meta:
                    try:
                        _test = torch.zeros(1)
                        _in_meta = (_test.device.type == 'meta')
                    except Exception:
                        _in_meta = False
                if _in_meta:
                    inv_freq = torch.empty(dim // 2, device='meta')
                    self.register_buffer("inv_freq", inv_freq, persistent=False)
                    cos_cached = torch.empty(max_position_embeddings, dim, device='meta')
                    sin_cached = torch.empty(max_position_embeddings, dim, device='meta')
                    self.register_buffer("cos_cached", cos_cached, persistent=False)
                    self.register_buffer("sin_cached", sin_cached, persistent=False)
                    self.max_seq_len_cached = max_position_embeddings
                    self._meta_init = True
                    return
                self._meta_init = False
                inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device or 'cpu') / self.dim))
                self.register_buffer("inv_freq", inv_freq, persistent=False)
                self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device,
                                        dtype=torch.get_default_dtype())
            def _set_cos_sin_cache(self, seq_len, device, dtype):
                self.max_seq_len_cached = seq_len
                t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
                freqs = torch.outer(t, self.inv_freq)
                emb = torch.cat((freqs, freqs), dim=-1)
                self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
                self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
            def _rebuild_cache_if_needed(self, device):
                """Rebuild cache after moving from meta device to real device."""
                if getattr(self, '_meta_init', False) and device.type != 'meta':
                    self._meta_init = False
                    inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
                    self.register_buffer("inv_freq", inv_freq, persistent=False)
                    self._set_cos_sin_cache(self.max_position_embeddings, device, torch.get_default_dtype())
            def forward(self, x, seq_len=None):
                self._rebuild_cache_if_needed(x.device)
                if seq_len is not None and seq_len > self.max_seq_len_cached:
                    self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
                return (self.cos_cached[:seq_len].to(dtype=x.dtype),
                        self.sin_cached[:seq_len].to(dtype=x.dtype))

        class _LlamaLinearScalingRotaryEmbedding(_LlamaRotaryEmbeddingOld):
            def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
                self.scaling_factor = scaling_factor
                super().__init__(dim, max_position_embeddings, base, device)
            def _set_cos_sin_cache(self, seq_len, device, dtype):
                if getattr(self, '_meta_init', False):
                    return
                self.max_seq_len_cached = seq_len
                t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype) / self.scaling_factor
                freqs = torch.outer(t, self.inv_freq)
                emb = torch.cat((freqs, freqs), dim=-1)
                self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
                self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

        class _LlamaDynamicNTKScalingRotaryEmbedding(_LlamaRotaryEmbeddingOld):
            def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
                self.scaling_factor = scaling_factor
                super().__init__(dim, max_position_embeddings, base, device)
            def _set_cos_sin_cache(self, seq_len, device, dtype):
                if getattr(self, '_meta_init', False):
                    return
                self.max_seq_len_cached = seq_len
                if seq_len > self.max_position_embeddings:
                    base = self.base * ((self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)) ** (self.dim / (self.dim - 2))
                    inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
                    self.register_buffer("inv_freq", inv_freq, persistent=False)
                t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
                freqs = torch.outer(t, self.inv_freq)
                emb = torch.cat((freqs, freqs), dim=-1)
                self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
                self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

        # Dispatching class that handles BOTH old and new APIs
        class _LlamaRotaryEmbedding(nn.Module):
            """Dispatches to old or new LlamaRotaryEmbedding based on call signature."""
            def __new__(cls, *args, **kwargs):
                # New API: config= keyword (transformers 5.x built-in code)
                if 'config' in kwargs:
                    config = kwargs['config']
                    # If rope_parameters is None (because we set rope_scaling=None
                    # for custom code compat), create old-style embedding instead
                    rope_params = getattr(config, 'rope_parameters', None)
                    if rope_params is None:
                        dim = config.hidden_size // config.num_attention_heads
                        max_pos = getattr(config, 'max_position_embeddings', 2048)
                        base = getattr(config, 'rope_theta', 10000.0)
                        return _LlamaRotaryEmbeddingOld(dim, max_pos, base)
                    if _real_LlamaRotaryEmbedding is not None:
                        return _real_LlamaRotaryEmbedding(*args, **kwargs)
                # Old API: (dim, max_pos, base, device) — custom modeling_llama2.py
                return _LlamaRotaryEmbeddingOld(*args, **kwargs)

        _real_apply_rotary_pos_emb = getattr(llama_module, 'apply_rotary_pos_emb', None)
        
        def _apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
            """Backward-compatible apply_rotary_pos_emb, handles both APIs.
            Old: (q, k, cos, sin, position_ids) where cos/sin need indexing
            New: (q, k, cos, sin, unsqueeze_dim=1) where cos/sin are pre-indexed
            """
            if position_ids is not None:
                # Old API: cos/sin shape is [seq_len, dim], need to index by position_ids
                cos = cos[position_ids].unsqueeze(1)
                sin = sin[position_ids].unsqueeze(1)
                from transformers.models.llama.modeling_llama import rotate_half
                q_embed = (q * cos) + (rotate_half(q) * sin)
                k_embed = (k * cos) + (rotate_half(k) * sin)
                return q_embed, k_embed
            else:
                # New API: delegate to real implementation
                if _real_apply_rotary_pos_emb is not None:
                    return _real_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
                # Fallback 
                cos = cos.unsqueeze(unsqueeze_dim)
                sin = sin.unsqueeze(unsqueeze_dim)
                from transformers.models.llama.modeling_llama import rotate_half
                q_embed = (q * cos) + (rotate_half(q) * sin)
                k_embed = (k * cos) + (rotate_half(k) * sin)
                return q_embed, k_embed

        # Inject all shims into the llama module's namespace
        # For rotary embeddings, FORCE replace — new transformers 5.x versions have
        # incompatible APIs (no max_position_embeddings arg)
        shims = {
            'Cache': Cache,
            'BaseModelOutputWithPast': BaseModelOutputWithPast,
            'CausalLMOutputWithPast': CausalLMOutputWithPast,
        }
        force_shims = {
            'LlamaRotaryEmbedding': _LlamaRotaryEmbedding,
            'LlamaLinearScalingRotaryEmbedding': _LlamaLinearScalingRotaryEmbedding,
            'LlamaDynamicNTKScalingRotaryEmbedding': _LlamaDynamicNTKScalingRotaryEmbedding,
            'apply_rotary_pos_emb': _apply_rotary_pos_emb,
        }
        
        injected = []
        for name, obj in shims.items():
            if not hasattr(llama_module, name):
                setattr(llama_module, name, obj)
                injected.append(name)
        for name, obj in force_shims.items():
            setattr(llama_module, name, obj)
            injected.append(name)
        
        # CRITICAL: Also add to __all__ so star imports pick them up
        all_names = list(shims.keys()) + list(force_shims.keys())
        if hasattr(llama_module, '__all__'):
            # NUCLEAR: delete __all__ entirely so `from ... import *` gets
            # everything from the namespace.  The custom modeling_llama2.py
            # relies on many names (LlamaMLP, LlamaRMSNorm, repeat_kv, etc.)
            # that transformers 5.x no longer lists in __all__.
            del llama_module.__all__
        
        if injected:
            print(f"Injected into llama module: {', '.join(injected)}")
        else:
            print("All shim classes already present in llama module")
    except ImportError as e:
        print(f"Warning: Could not inject shims into llama module: {e}")


import contextlib

@contextlib.contextmanager
def _meta_safe_loading():
    """
    Context manager that monkey-patches torch.Tensor.item() to handle meta tensors.
    
    In transformers 5.x, models are always created on meta device during __init__.
    InternVL-based models (ArtiMuse, Sa2VA) call torch.linspace().item() in __init__
    which fails on meta tensors. This patch returns 0.0 for meta tensor .item() calls,
    which is safe because the actual values are set later during weight materialization.
    """
    _original_item = torch.Tensor.item
    
    def _safe_item(self):
        if self.device.type == 'meta':
            return 0.0
        return _original_item(self)
    
    torch.Tensor.item = _safe_item
    try:
        yield
    finally:
        torch.Tensor.item = _original_item


class QInsightHandler:
    """
    Handler for Q-Insight (OneAlign) specialist model for image quality assessment.
    Model: q-future/one-align (EXACT from I2I-Bench paper Section A.2)
    Used for: Overall Image Quality dimension.
    
    Loads in-process with compatibility patches for transformers 5.x.
    The model's custom code requires patching for Cache import compatibility.
    """
    
    _instance = None
    _model = None
    
    def __new__(cls, config: EvaluatorConfig):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: EvaluatorConfig):
        self.config = config
        self._initialized = False
    
    def initialize(self):
        """Lazy initialization with compatibility patches."""
        if self._initialized:
            return
        
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers library is required for Q-Insight.")
        
        print(f"Loading Q-Insight model: {self.config.q_insight_model_name}")
        
        # Pre-download custom code to modules cache so patches can find the files
        # AutoConfig downloads config + custom code without instantiating the model
        q_config = None
        try:
            q_config = AutoConfig.from_pretrained(self.config.q_insight_model_name, trust_remote_code=True)
            print(f"  Config loaded: {type(q_config).__name__}")
        except Exception as e:
            print(f"Config pre-download note (non-fatal): {e}")
        
        # Apply compatibility patches before loading
        _inject_cache_into_llama_module()
        _patch_q_insight_model_files()
        
        # ===================================================================
        # RUNTIME FIX: Ensure rope_scaling dicts have "type" key (transformers
        # 5.x renames "type" → "rope_type", but the custom modeling_llama2.py
        # code expects the old "type" key).
        # ===================================================================
        def _fix_rope_scaling(cfg):
            """Recursively fix rope_scaling dicts in a config tree.
            
            In transformers 5.x, rope_scaling={"rope_type": "default"} replaces
            the old rope_scaling=None. The custom modeling_llama2.py code expects
            None for default behavior, so we convert back.
            """
            if hasattr(cfg, 'rope_scaling') and isinstance(cfg.rope_scaling, dict):
                rope_type = cfg.rope_scaling.get('rope_type', cfg.rope_scaling.get('type', ''))
                if rope_type == 'default':
                    # Old code expects None for default rope — convert back
                    cfg.rope_scaling = None
                    print(f"  Fixed rope_scaling on {type(cfg).__name__}: set to None (was default)")
                elif 'type' not in cfg.rope_scaling:
                    # Non-default but missing old "type" key — add it
                    cfg.rope_scaling['type'] = rope_type
                    if 'factor' not in cfg.rope_scaling:
                        cfg.rope_scaling['factor'] = 1.0
                    print(f"  Fixed rope_scaling on {type(cfg).__name__}: added 'type'={rope_type}")
            # Walk nested config objects
            for attr_name in dir(cfg):
                if attr_name.startswith('_'):
                    continue
                try:
                    attr_val = getattr(cfg, attr_name)
                except Exception:
                    continue
                if hasattr(attr_val, 'rope_scaling') and attr_val is not cfg:
                    _fix_rope_scaling(attr_val)
        
        if q_config is not None:
            _fix_rope_scaling(q_config)
        
        # ===================================================================
        # RUNTIME FIX: Add missing config attributes for transformers 5.x compat.
        # The custom MPLUGOwl2Config/LlamaConfig doesn't have mlp_bias,
        # attention_dropout etc. that transformers 5.x code expects.
        # ===================================================================
        def _fix_config_attrs(cfg):
            """Recursively add missing config attributes."""
            defaults = {
                'mlp_bias': False,
                'attention_dropout': 0.0,
                'attention_bias': False,
            }
            fixed = []
            for attr, default_val in defaults.items():
                if not hasattr(cfg, attr):
                    setattr(cfg, attr, default_val)
                    fixed.append(attr)
            if fixed:
                print(f"  Added missing config attrs on {type(cfg).__name__}: {', '.join(fixed)}")
            # Walk nested config objects  
            for attr_name in ['text_config', 'vision_config']:
                if hasattr(cfg, attr_name):
                    sub = getattr(cfg, attr_name)
                    if sub is not None and sub is not cfg:
                        _fix_config_attrs(sub)
        
        if q_config is not None:
            _fix_config_attrs(q_config)
        
        # Also inject BaseModelOutputWithPast into llama module
        try:
            from transformers.modeling_outputs import BaseModelOutputWithPast
            import transformers.models.llama.modeling_llama as llama_module
            if not hasattr(llama_module, 'BaseModelOutputWithPast'):
                llama_module.BaseModelOutputWithPast = BaseModelOutputWithPast
        except ImportError:
            pass
        
        # Clear any cached modules that might have stale imports
        import importlib
        keys_to_remove = [k for k in sys.modules.keys() 
                         if 'q_hyphen_future' in k or 'one_hyphen_align' in k 
                         or 'q-future' in k or 'one-align' in k
                         or 'mplug_owl2' in k]
        for k in keys_to_remove:
            del sys.modules[k]
            print(f"  Invalidated cached module: {k}")
        
        # PRE-EMPTIVE: inject logger into builtins BEFORE from_pretrained so any 
        # module imported during model loading can resolve 'logger' via builtins fallback
        import builtins
        import logging as _logging
        if not hasattr(builtins, 'logger'):
            builtins.logger = _logging.getLogger('q_insight_global')
            print("  Injected logger into builtins (pre-loading)")
        
        # Pass the fixed config so from_pretrained uses our patched rope_scaling
        model_kwargs = dict(
            trust_remote_code=True,
            attn_implementation="eager",
            torch_dtype=torch.float16,
            device_map="auto",
        )
        if q_config is not None:
            model_kwargs["config"] = q_config
        
        with _meta_safe_loading():
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.q_insight_model_name,
                **model_kwargs,
            )
        self._model.eval()
        
        # Inject 'logger' into ALL modules that might be called by score()
        # The q-future/one-align score() method references 'logger' which may not be defined
        import logging as _logging
        
        # 1. Inject into the model class's module
        model_module_name = type(self._model).__module__
        if model_module_name in sys.modules:
            model_mod = sys.modules[model_module_name]
            if not hasattr(model_mod, 'logger'):
                model_mod.logger = _logging.getLogger(model_module_name)
                print(f"  Injected logger into module: {model_module_name}")
        
        # 2. Inject directly into score() method's __globals__ (most robust)
        if hasattr(self._model, 'score'):
            score_fn = getattr(self._model.score, '__func__', self._model.score)
            if hasattr(score_fn, '__globals__') and 'logger' not in score_fn.__globals__:
                score_fn.__globals__['logger'] = _logging.getLogger('q_insight_score')
                print(f"  Injected logger into score().__globals__")
        
        # 3. Inject into ALL q-future/one-align/mplug related modules in sys.modules
        for mod_name, mod_obj in list(sys.modules.items()):
            if any(pat in mod_name for pat in ['q_hyphen_future', 'one_hyphen_align', 
                                                'q-future', 'one-align', 'mplug_owl2']):
                if mod_obj is not None and not hasattr(mod_obj, 'logger'):
                    mod_obj.logger = _logging.getLogger(mod_name)
                    print(f"  Injected logger into related module: {mod_name}")
        
        self._initialized = True
        print("Q-Insight (OneAlign) loaded successfully.")
    
    def get_quality_score(self, image: Image.Image) -> float:
        """
        Get overall image quality score from Q-Insight (OneAlign).
        Uses model.score() API which returns a score in [1, 5] range.
        
        Returns:
            Quality score normalized to [0, 1]
        """
        self.initialize()
        
        # Nuclear option: inject logger into builtins so it's available in ALL modules
        # This resolves NameError for 'logger' in any q-future custom code
        import builtins
        import logging as _logging
        _had_logger = hasattr(builtins, 'logger')
        if not _had_logger:
            builtins.logger = _logging.getLogger('q_insight_builtins')
        
        try:
            # The OneAlign model has a custom .score() method
            score = self._model.score([image], task_="quality", input_="image")
        finally:
            # Clean up builtins to avoid side effects
            if not _had_logger and hasattr(builtins, 'logger'):
                del builtins.logger
        
        # Convert to float
        if hasattr(score, 'item'):
            score = score.item()
        elif isinstance(score, (list, tuple)):
            score = float(score[0])
        else:
            score = float(score)
        
        # OneAlign returns scores in [1, 5] range, normalize to [0, 1]
        normalized_score = (score - 1.0) / 4.0
        return max(0.0, min(1.0, normalized_score))


class ArtiMuseHandler:
    """
    Handler for ArtiMuse specialist model for aesthetic quality assessment.
    Model: Thunderbolt215215/ArtiMuse (EXACT from I2I-Bench paper Section A.2)
    Used for: Aesthetic Quality dimension.
    
    ArtiMuse is an InternVL3-based model with a specialized .score() method
    that uses 101 aesthetic tokens (0-100) and softmax over logits.
    NO FALLBACKS - raises error if model cannot be loaded.
    """
    
    _instance = None
    _model = None
    _tokenizer = None
    
    def __new__(cls, config: EvaluatorConfig):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: EvaluatorConfig):
        self.config = config
        self._initialized = False
    
    def initialize(self):
        """Lazy initialization. Raises error on failure."""
        if self._initialized:
            return
        
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers library is required for ArtiMuse.")
        
        print(f"Loading ArtiMuse model: {self.config.artimuse_model_name}")
        
        # Pre-download custom code to modules cache so patches can find the files
        try:
            AutoConfig.from_pretrained(self.config.artimuse_model_name, trust_remote_code=True)
        except Exception as e:
            print(f"Config pre-download note (non-fatal): {e}")
        
        # Patch InternVL cached files to fix meta tensor issues with transformers 5.x
        _patch_internvl_meta_tensor_issue()
        
        # Patch ArtiMuse to add score() and generate_logits() methods (file-based)
        _patch_artimuse_score_method()
        
        # CRITICAL: Invalidate cached ArtiMuse modules so re-import picks up patched files
        import sys
        keys_to_remove = [k for k in sys.modules if 'ArtiMuse' in k or 'artimuse' in k.lower()
                         or 'Thunderbolt215215' in k]
        for k in keys_to_remove:
            del sys.modules[k]
            print(f"  Invalidated cached module: {k}")
        
        # ArtiMuse uses InternVL architecture with trust_remote_code
        # Use _meta_safe_loading to handle torch.linspace().item() on meta device
        with _meta_safe_loading():
            try:
                self._model = AutoModel.from_pretrained(
                    self.config.artimuse_model_name,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    use_flash_attn=True
                ).eval().cuda()
            except Exception as e:
                print(f"Flash attention failed ({e}), falling back to eager attention")
                self._model = AutoModel.from_pretrained(
                    self.config.artimuse_model_name,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    use_flash_attn=False
                ).eval().cuda()
        
        # Dynamically bind score() and generate_logits() if not present
        if not hasattr(self._model, 'score'):
            print("  Binding score() and generate_logits() methods dynamically...")
            _bind_artimuse_score_methods(self._model)
        
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.artimuse_model_name,
            trust_remote_code=True,
            use_fast=False
        )
        self._device = "cuda"
        self._initialized = True
        print("ArtiMuse loaded successfully.")
    
    def get_aesthetic_score(self, image: Image.Image) -> float:
        """
        Get aesthetic quality score from ArtiMuse using its native .score() method.
        
        ArtiMuse's score() method:
        1. Uses a special prompt mapping scores 0-100 to two-letter aesthetic tokens
        2. Gets logits via generate_logits()
        3. Applies softmax over 101 aesthetic token logits
        4. Computes weighted sum: softmax(logits) @ [0, 1, ..., 100]
        5. Returns a score in [0, 100]
        
        Returns:
            Aesthetic score normalized to [0, 1]
        """
        self.initialize()
        
        # Prepare image - ArtiMuse expects 448x448 normalized tensors
        pixel_values = self._load_image(image)
        
        generation_config = dict(
            max_new_tokens=8192,
            do_sample=False,
            pad_token_id=self._tokenizer.eos_token_id
        )
        
        # Use ArtiMuse's built-in score() method which returns 0-100
        score = self._model.score(
            self._device,
            self._tokenizer,
            pixel_values,
            generation_config
        )
        
        # Convert to float if needed
        if hasattr(score, 'item'):
            score = score.item()
        else:
            score = float(score)
        
        # Normalize from [0, 100] to [0, 1]
        normalized_score = score / 100.0
        return max(0.0, min(1.0, normalized_score))
    
    def _load_image(self, image: Image.Image):
        """Load and preprocess image for ArtiMuse (448x448, ImageNet normalized)."""
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
        pixel_values = transform(image).unsqueeze(0).to(torch.bfloat16).to(self._device)
        return pixel_values


class GOTOCRHandler:
    """
    Handler for GOT-OCR2.0 specialist model for text extraction.
    Model: stepfun-ai/GOT-OCR-2.0-hf (EXACT from I2I-Bench paper Section A.2)
    Used for: Text Content & Style Accuracy dimension (content part).
    NO FALLBACKS - raises error if model cannot be loaded.
    """
    
    _instance = None
    _model = None
    _processor = None
    
    def __new__(cls, config: EvaluatorConfig):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: EvaluatorConfig):
        self.config = config
        self._initialized = False
    
    def initialize(self):
        """Lazy initialization. Raises error on failure."""
        if self._initialized:
            return
        
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers library is required for GOT-OCR.")
        
        print(f"Loading GOT-OCR2.0 model: {self.config.got_ocr_model_name}")
        # Use AutoModelForImageTextToText for vision-to-text generation
        from transformers import AutoModelForImageTextToText
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.config.got_ocr_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self._processor = AutoProcessor.from_pretrained(
            self.config.got_ocr_model_name,
            trust_remote_code=True
        )
        self._model.eval()
        self._initialized = True
        print("GOT-OCR2.0 loaded successfully.")
    
    def extract_text(self, image: Image.Image) -> str:
        """
        Extract text from image using GOT-OCR2.0.
        Uses the HuggingFace API as documented on the model page.
        
        Returns:
            Extracted text string
        """
        self.initialize()
        
        # Prepare inputs using the processor (plain text OCR mode)
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        
        # Generate text using the documented API
        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                do_sample=False,
                tokenizer=self._processor.tokenizer,
                stop_strings="<|im_end|>",
                max_new_tokens=512
            )
        
        # Decode - skip the input tokens
        result = self._processor.decode(
            generated_ids[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return result.strip()
    
    def compute_content_accuracy(self, extracted_text: str, target_text: str) -> float:
        """
        Compute content accuracy using Levenshtein distance.
        As specified in the I2I-Bench paper.
        
        Returns:
            Content accuracy score Sc in [0, 1]
        """
        if not target_text:
            return 1.0 if not extracted_text else 0.0
        
        if not extracted_text:
            return 0.0
        
        # Levenshtein distance
        distance = self._levenshtein_distance(
            extracted_text.lower(), 
            target_text.lower()
        )
        max_len = max(len(extracted_text), len(target_text))
        
        accuracy = 1.0 - (distance / max_len) if max_len > 0 else 1.0
        return max(0.0, min(1.0, accuracy))
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Compute Levenshtein edit distance."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]


class DINOv3Handler:
    """
    Handler for DINOv3 specialist model for feature extraction.
    Model: facebook/dinov3-vitl16-pretrain-lvd1689m (EXACT from I2I-Bench paper Section A.2)
    Used for: Non-Edited Region Fidelity, Subject Identity Fidelity dimensions.
    NO FALLBACKS - raises error if model cannot be loaded.
    """
    
    _instance = None
    _model = None
    _processor = None
    
    def __new__(cls, config: EvaluatorConfig):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: EvaluatorConfig):
        self.config = config
        self._initialized = False
    
    def initialize(self):
        """Lazy initialization. Raises error on failure."""
        if self._initialized:
            return
        
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers library is required for DINOv3.")
        
        print(f"Loading DINOv3 model: {self.config.dinov3_model_name}")
        
        # DINOv3 uses AutoImageProcessor (not AutoProcessor) per the HF model card
        from transformers import AutoImageProcessor
        
        # Pre-download custom code to modules cache
        try:
            AutoConfig.from_pretrained(self.config.dinov3_model_name, trust_remote_code=True)
        except Exception as e:
            print(f"Config pre-download note (non-fatal): {e}")
        
        # Load without device_map='auto' to avoid meta tensor issues in transformers 5.x
        with _meta_safe_loading():
            self._model = AutoModel.from_pretrained(
                self.config.dinov3_model_name,
                torch_dtype=torch.float32,  # DINOv3 typically uses float32
                trust_remote_code=True
            ).eval().cuda()
        self._processor = AutoImageProcessor.from_pretrained(
            self.config.dinov3_model_name,
            trust_remote_code=True
        )
        self._initialized = True
        print("DINOv3 loaded successfully.")
    
    def extract_features(self, image: Image.Image, 
                         mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extract features from image region defined by mask using DINOv3.
        
        Args:
            image: Input image
            mask: Optional binary mask (1 for region of interest)
            
        Returns:
            Feature vector
        """
        self.initialize()
        
        # If mask is provided, apply it to the image
        if mask is not None:
            # Create masked image
            img_array = np.array(image)
            mask_3d = np.stack([mask] * 3, axis=-1) if len(mask.shape) == 2 else mask
            masked_array = img_array * mask_3d
            image = Image.fromarray(masked_array.astype(np.uint8))
        
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._model(**inputs)
            # Use CLS token features or mean pooled features
            if hasattr(outputs, 'last_hidden_state'):
                features = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]  # CLS token
            else:
                features = outputs.pooler_output.cpu().numpy()[0]
        
        return features
    
    def compute_similarity(self, features1: np.ndarray, 
                          features2: np.ndarray) -> float:
        """
        Compute cosine similarity between two feature vectors.
        As per Eq. (2) in the I2I-Bench paper.
        
        Returns:
            Cosine similarity score in [0, 1]
        """
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(features1, features2) / (norm1 * norm2)
        # Cosine similarity is in [-1, 1], map to [0, 1]
        return max(0.0, min(1.0, (similarity + 1) / 2))


class Sa2VAHandler:
    """
    Handler for Sa2VA segmentation model.
    Model: ByteDance/Sa2VA-4B (EXACT from I2I-Bench paper Section A.2)
    Used for: Non-Edited Region Fidelity, Subject Identity Fidelity dimensions.
    NO FALLBACKS - raises error if model cannot be loaded.
    """
    
    _instance = None
    _model = None
    _tokenizer = None
    
    def __new__(cls, config: EvaluatorConfig):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: EvaluatorConfig):
        self.config = config
        self._initialized = False
    
    def initialize(self):
        """Lazy initialization. Raises error on failure."""
        if self._initialized:
            return
        
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers library is required for Sa2VA.")
        
        print(f"Loading Sa2VA model: {self.config.sa2va_model_name}")
        # Pre-download custom code to modules cache so patches can find the files
        try:
            AutoConfig.from_pretrained(self.config.sa2va_model_name, trust_remote_code=True)
        except Exception as e:
            print(f"Config pre-download note (non-fatal): {e}")
        # Patch InternVL cached files to fix meta tensor issues with transformers 5.x
        _patch_internvl_meta_tensor_issue()
        # Sa2VA is a specialized segmentation model based on InternVL
        # Load with eager attention (flash_attn not available)
        # Use _meta_safe_loading to handle torch.linspace().item() on meta device
        with _meta_safe_loading():
            self._model = AutoModel.from_pretrained(
                self.config.sa2va_model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False,  # Avoid meta tensors
                use_flash_attn=False
            ).eval().cuda()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.sa2va_model_name,
            trust_remote_code=True,
            use_fast=False
        )
        self._initialized = True
        print("Sa2VA loaded successfully.")
    
    def segment(self, image: Image.Image, prompt: str) -> np.ndarray:
        """
        Generate segmentation mask based on text prompt using Sa2VA.
        
        Args:
            image: Input image
            prompt: Text prompt describing what to segment (e.g., "Please segment the horse")
            
        Returns:
            Binary mask as numpy array (1 for segmented region)
        """
        self.initialize()
        
        # Sa2VA uses predict_forward method for segmentation
        # Format: "<image>Please segment the [object]."
        text_prompts = f"<image>{prompt}"
        
        input_dict = {
            'image': image,
            'text': text_prompts,
            'past_text': '',
            'mask_prompts': None,
            'tokenizer': self._tokenizer,
        }
        
        try:
            return_dict = self._model.predict_forward(**input_dict)
            masks = return_dict.get('prediction_masks', None)
            
            if masks and len(masks) > 0:
                # Return first mask, normalize to [0, 1]
                mask = masks[0]
                if mask.ndim == 3:  # (1, h, w)
                    mask = mask[0]
                return mask.astype(np.float32)
            else:
                # No mask found, return full image mask (conservative)
                w, h = image.size
                return np.ones((h, w), dtype=np.float32)
        except Exception as e:
            print(f"Warning: Sa2VA segmentation failed: {e}")
            # Return full image mask as fallback
            w, h = image.size
            return np.ones((h, w), dtype=np.float32)
    
    def _load_image(self, image: Image.Image):
        """Load and preprocess image for Sa2VA."""
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        
        def build_transform(input_size):
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
            return transform
        
        transform = build_transform(448)
        pixel_values = transform(image).unsqueeze(0).to(torch.bfloat16).to(self._model.device)
        return pixel_values
    
    def _parse_mask(self, response: str, image_size: Tuple[int, int]) -> np.ndarray:
        """
        Parse segmentation mask from Sa2VA response.
        Sa2VA outputs masks in various formats depending on the prompt.
        """
        w, h = image_size
        
        # Sa2VA can return masks as coordinates, polygons, or base64 encoded masks
        # Try to parse different formats
        
        # Check for polygon coordinates in response
        import re
        coord_pattern = r'\[(\d+),\s*(\d+)\]'
        coords = re.findall(coord_pattern, response)
        
        if coords:
            # Create mask from polygon coordinates
            mask = np.zeros((h, w), dtype=np.float32)
            points = [(int(x), int(y)) for x, y in coords]
            
            if len(points) >= 3:
                # Fill polygon
                from PIL import ImageDraw
                mask_img = Image.new('L', (w, h), 0)
                draw = ImageDraw.Draw(mask_img)
                draw.polygon(points, fill=255)
                mask = np.array(mask_img) / 255.0
        else:
            # If no coordinates found, return a default mask (full image)
            # This handles cases where segmentation fails
            mask = np.ones((h, w), dtype=np.float32)
        
        return mask


# ==============================================================================
# EVALUATION DIMENSIONS - 7 COMMON DIMENSIONS
# ==============================================================================

class OverallImageQualityEvaluator(BaseEvaluator):
    """
    Dimension 1: Overall Image Quality
    Pipeline: Specialist Model (Q-Insight)
    
    Evaluates the objective technical quality (e.g., clarity, noise) of the image.
    """
    
    def __init__(self, config: EvaluatorConfig):
        super().__init__(config)
        self.q_insight = QInsightHandler(config)
    
    @property
    def dimension_name(self) -> str:
        return "Overall_Image_Quality"
    
    def evaluate(self, original_image: Image.Image, edited_image: Image.Image,
                 instruction: str, **kwargs) -> Dict[str, Any]:
        """
        Evaluate overall image quality using Q-Insight.
        
        Returns:
            Dictionary with 'score' (0-1) and 'details'
        """
        score = self.q_insight.get_quality_score(edited_image)
        
        return {
            "score": score,
            "details": {
                "method": "Q-Insight Specialist",
                "description": "Objective technical quality assessment"
            }
        }


class AestheticQualityEvaluator(BaseEvaluator):
    """
    Dimension 2: Aesthetic Quality
    Pipeline: Specialist Model (ArtiMuse)
    
    Evaluates the artistic appeal of the image.
    """
    
    def __init__(self, config: EvaluatorConfig):
        super().__init__(config)
        self.artimuse = ArtiMuseHandler(config)
    
    @property
    def dimension_name(self) -> str:
        return "Aesthetic_Quality"
    
    def evaluate(self, original_image: Image.Image, edited_image: Image.Image,
                 instruction: str, **kwargs) -> Dict[str, Any]:
        """
        Evaluate aesthetic quality using ArtiMuse.
        
        Returns:
            Dictionary with 'score' (0-1) and 'details'
        """
        score = self.artimuse.get_aesthetic_score(edited_image)
        
        return {
            "score": score,
            "details": {
                "method": "ArtiMuse Specialist",
                "description": "Artistic and aesthetic quality assessment"
            }
        }


class BlendingNaturalnessEvaluator(BaseEvaluator):
    """
    Dimension 3: Blending Naturalness
    Pipeline: LMM VQA-5Level
    
    Evaluates the smoothness of transition between edited and surrounding regions.
    """
    
    def __init__(self, config: EvaluatorConfig):
        super().__init__(config)
        self.lmm = LMMHandler(config)
    
    @property
    def dimension_name(self) -> str:
        return "Blending_Naturalness"
    
    def evaluate(self, original_image: Image.Image, edited_image: Image.Image,
                 instruction: str, **kwargs) -> Dict[str, Any]:
        """
        Evaluate blending naturalness using LMM VQA-5Level.
        Uses Eq. (1) from the paper.
        
        Returns:
            Dictionary with 'score' (0-1) and 'details'
        """
        choices = ["excellent", "good", "fair", "poor", "bad"]
        weights = self.config.vqa_5level_weights
        
        probs = self.lmm.get_logits_for_choices(
            images=[original_image, edited_image],
            system_prompt=Prompts.BLENDING_NATURALNESS_SYSTEM,
            user_prompt=Prompts.BLENDING_NATURALNESS_USER,
            choices=choices
        )
        
        # Compute weighted score as per Eq. (1)
        score = sum(weights[c] * probs[c] for c in choices)
        
        return {
            "score": score,
            "details": {
                "method": "LMM VQA-5Level",
                "probabilities": probs,
                "response": max(probs, key=probs.get)
            }
        }


class EditingArtifactsEvaluator(BaseEvaluator):
    """
    Dimension 4: Generative/Editing Artifacts
    Pipeline: LMM VQA-5Level
    
    Evaluates the introduction of AI-specific flaws (e.g., broken structures, unnatural textures).
    """
    
    def __init__(self, config: EvaluatorConfig):
        super().__init__(config)
        self.lmm = LMMHandler(config)
    
    @property
    def dimension_name(self) -> str:
        return "Editing_Artifacts"
    
    def evaluate(self, original_image: Image.Image, edited_image: Image.Image,
                 instruction: str, **kwargs) -> Dict[str, Any]:
        """
        Evaluate editing artifacts using LMM VQA-5Level.
        
        Returns:
            Dictionary with 'score' (0-1) and 'details'
        """
        choices = ["excellent", "good", "fair", "poor", "bad"]
        weights = self.config.vqa_5level_weights
        
        probs = self.lmm.get_logits_for_choices(
            images=[original_image, edited_image],
            system_prompt=Prompts.EDITING_ARTIFACTS_SYSTEM,
            user_prompt=Prompts.EDITING_ARTIFACTS_USER,
            choices=choices
        )
        
        score = sum(weights[c] * probs[c] for c in choices)
        
        return {
            "score": score,
            "details": {
                "method": "LMM VQA-5Level",
                "probabilities": probs,
                "response": max(probs, key=probs.get)
            }
        }


class InstructionFollowingMacroEvaluator(BaseEvaluator):
    """
    Dimension 5: Instruction Following (Macro)
    Pipeline: LMM VQA-5Level
    
    Assesses the model's overall understanding and execution of the instruction's core intent.
    """
    
    def __init__(self, config: EvaluatorConfig):
        super().__init__(config)
        self.lmm = LMMHandler(config)
    
    @property
    def dimension_name(self) -> str:
        return "Instruction_Following_Macro"
    
    def evaluate(self, original_image: Image.Image, edited_image: Image.Image,
                 instruction: str, **kwargs) -> Dict[str, Any]:
        """
        Evaluate instruction following using LMM VQA-5Level.
        
        Returns:
            Dictionary with 'score' (0-1) and 'details'
        """
        choices = ["excellent", "good", "fair", "poor", "bad"]
        weights = self.config.vqa_5level_weights
        
        user_prompt = Prompts.INSTRUCTION_FOLLOWING_USER.format(prompt=instruction)
        
        probs = self.lmm.get_logits_for_choices(
            images=[original_image, edited_image],
            system_prompt=Prompts.INSTRUCTION_FOLLOWING_SYSTEM,
            user_prompt=user_prompt,
            choices=choices
        )
        
        score = sum(weights[c] * probs[c] for c in choices)
        
        return {
            "score": score,
            "details": {
                "method": "LMM VQA-5Level",
                "probabilities": probs,
                "response": max(probs, key=probs.get),
                "instruction": instruction
            }
        }


class NonEditedRegionFidelityEvaluator(BaseEvaluator):
    """
    Dimension 6: Non-Edited Region Fidelity
    Pipeline: Hybrid LMM-Specialist (Sa2VA + DINOv3)
    
    Assesses whether image regions not targeted by the instruction remain unchanged.
    Uses Eq. (2) from the paper.
    """
    
    def __init__(self, config: EvaluatorConfig):
        super().__init__(config)
        # Don't initialize handlers here - we'll do it on-demand with memory management
        self.config = config
    
    @property
    def dimension_name(self) -> str:
        return "Non_Edited_Region_Fidelity"
    
    def evaluate(self, original_image: Image.Image, edited_image: Image.Image,
                 instruction: str, **kwargs) -> Dict[str, Any]:
        """
        Evaluate non-edited region fidelity using Hybrid LMM-Specialist pipeline.
        
        Process:
        1. LMM generates segmentation command for edited region
        2. Sa2VA segments the edited region (Medit)
        3. Invert mask to get non-edited region (Mnon-edit)
        4. DINOv3 extracts features from non-edited regions
        5. Compute cosine similarity (Eq. 2)
        
        Returns:
            Dictionary with 'score' (0-1) and 'details'
        
        Note: This method carefully manages GPU memory by loading/unloading
        models sequentially since LMM (~16GB) + Sa2VA (~14GB) > 24GB GPU.
        """
        # Step 1: Generate segmentation command with LMM
        seg_prompt = Prompts.NON_EDITED_REGION_SEGMENTATION.format(
            edit_instruction=instruction
        )
        
        lmm = LMMHandler(self.config)
        seg_command = lmm.generate(
            images=[original_image, edited_image],
            system_prompt="You are a segmentation assistant.",
            user_prompt=seg_prompt,
            max_tokens=100
        )
        
        # Clear LMM from GPU before loading Sa2VA
        clear_gpu_memory()
        
        # Step 2: Segment edited region with Sa2VA
        sa2va = Sa2VAHandler(self.config)
        mask_edit = sa2va.segment(edited_image, seg_command)
        
        # Clear Sa2VA from GPU before loading DINOv3
        clear_gpu_memory()
        
        # Step 3: Invert mask to get non-edited region
        mask_non_edit = 1.0 - mask_edit
        
        # Resize mask to match original image size if needed
        orig_size = (original_image.width, original_image.height)
        edit_size = (edited_image.width, edited_image.height)
        
        # Resize masks to match respective image sizes
        if mask_non_edit.shape[:2] != (original_image.height, original_image.width):
            mask_for_orig = np.array(Image.fromarray((mask_non_edit * 255).astype(np.uint8)).resize(
                orig_size, Image.Resampling.NEAREST)) / 255.0
        else:
            mask_for_orig = mask_non_edit
            
        if mask_non_edit.shape[:2] != (edited_image.height, edited_image.width):
            mask_for_edit = np.array(Image.fromarray((mask_non_edit * 255).astype(np.uint8)).resize(
                edit_size, Image.Resampling.NEAREST)) / 255.0
        else:
            mask_for_edit = mask_non_edit
        
        # Step 4 & 5: Extract features and compute similarity with DINOv3
        dino = DINOv3Handler(self.config)
        features_orig = dino.extract_features(original_image, mask_for_orig)
        features_edit = dino.extract_features(edited_image, mask_for_edit)
        
        # Compute cosine similarity (Eq. 2)
        score = dino.compute_similarity(features_orig, features_edit)
        
        return {
            "score": score,
            "details": {
                "method": "Hybrid LMM-Specialist (Sa2VA + DINOv3)",
                "segmentation_command": seg_command,
                "description": "Cosine similarity of non-edited regions"
            }
        }


class PhysicalPlausibilityEvaluator(BaseEvaluator):
    """
    Dimension 7: Physical Plausibility
    Pipeline: LMM Multi-Question VQA (Multi-VQA)
    
    Evaluates whether the edit adheres to physical laws (lighting, perspective, gravity).
    Uses Eq. (3) from the paper.
    """
    
    def __init__(self, config: EvaluatorConfig):
        super().__init__(config)
        self.lmm = LMMHandler(config)
    
    @property
    def dimension_name(self) -> str:
        return "Physical_Plausibility"
    
    def evaluate(self, original_image: Image.Image, edited_image: Image.Image,
                 instruction: str, **kwargs) -> Dict[str, Any]:
        """
        Evaluate physical plausibility using LMM Multi-Question VQA.
        
        Two-stage process:
        1. Question Generation: LMM generates N Yes/No questions
        2. Question Answering: LMM answers each question
        Final score = ratio of 'Yes' answers (Eq. 3)
        
        Returns:
            Dictionary with 'score' (0-1) and 'details'
        """
        # Stage 1: Question Generation
        qg_prompt = Prompts.PHYSICAL_PLAUSIBILITY_QG.format(prompt=instruction)
        
        questions_response = self.lmm.generate(
            images=[original_image],
            system_prompt="You are an expert in physical plausibility evaluation.",
            user_prompt=qg_prompt,
            max_tokens=1024
        )
        
        # Parse questions from JSON response
        questions = self._parse_questions(questions_response, "physical_questions")
        
        if not questions:
            # Fallback questions
            questions = [
                "Is the lighting on the edited element consistent with the scene?",
                "Are the shadows in the edited image physically correct?",
                "Does the edited element have correct perspective?",
                "Is the scale of the edited element appropriate for the scene?",
                "Does the edited image obey basic physical laws?"
            ]
        
        # Stage 2: Question Answering
        yes_count = 0
        qa_results = []
        
        for q in questions:
            qa_prompt = Prompts.PHYSICAL_PLAUSIBILITY_QA.format(question=q)
            
            answer = self.lmm.generate(
                images=[original_image, edited_image],
                system_prompt="You are an expert evaluator. Answer Yes or No only.",
                user_prompt=qa_prompt,
                max_tokens=10
            )
            
            is_yes = "yes" in answer.lower()
            if is_yes:
                yes_count += 1
            
            qa_results.append({
                "question": q,
                "answer": answer,
                "is_yes": is_yes
            })
        
        # Compute score as ratio of Yes answers (Eq. 3)
        score = yes_count / len(questions) if questions else 0.0
        
        return {
            "score": score,
            "details": {
                "method": "LMM Multi-Question VQA",
                "num_questions": len(questions),
                "yes_count": yes_count,
                "qa_results": qa_results
            }
        }
    
    def _parse_questions(self, response: str, key: str) -> List[str]:
        """Parse questions from JSON response."""
        try:
            # Find JSON in response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data.get(key, [])
        except json.JSONDecodeError:
            pass
        
        # Try to extract questions directly
        questions = re.findall(r'Question \d+:?\s*([^\n?]+\?)', response)
        if questions:
            return questions
        
        return []


# ==============================================================================
# EVALUATION DIMENSIONS - 8 SPECIFIC DIMENSIONS
# ==============================================================================

class ObjectManipulationAccuracyEvaluator(BaseEvaluator):
    """
    Dimension 8: Object Manipulation Accuracy
    Pipeline: LMM VQA-2Level
    
    Evaluates the correctness of add/remove/replace object operations.
    Uses Eq. (4) from the paper.
    """
    
    def __init__(self, config: EvaluatorConfig):
        super().__init__(config)
        self.lmm = LMMHandler(config)
    
    @property
    def dimension_name(self) -> str:
        return "Object_Manipulation_Accuracy"
    
    def evaluate(self, original_image: Image.Image, edited_image: Image.Image,
                 instruction: str, **kwargs) -> Dict[str, Any]:
        """
        Evaluate object manipulation accuracy using LMM VQA-2Level.
        Score = P('Yes') as per Eq. (4)
        
        Returns:
            Dictionary with 'score' (0-1) and 'details'
        """
        user_prompt = Prompts.VQA_2LEVEL_USER.format(prompt=instruction)
        
        probs = self.lmm.get_logits_for_choices(
            images=[original_image, edited_image],
            system_prompt=Prompts.VQA_2LEVEL_SYSTEM,
            user_prompt=user_prompt,
            choices=["Yes", "No"]
        )
        
        # Score is P('Yes') as per Eq. (4)
        score = probs.get("Yes", 0.0)
        
        return {
            "score": score,
            "details": {
                "method": "LMM VQA-2Level",
                "probabilities": probs,
                "instruction": instruction
            }
        }


class LocalAttributeAccuracyEvaluator(BaseEvaluator):
    """
    Dimension 9: Local Attribute Accuracy
    Pipeline: LMM VQA-2Level
    
    Evaluates the accurate application of attribute modifications (change color/material).
    """
    
    def __init__(self, config: EvaluatorConfig):
        super().__init__(config)
        self.lmm = LMMHandler(config)
    
    @property
    def dimension_name(self) -> str:
        return "Local_Attribute_Accuracy"
    
    def evaluate(self, original_image: Image.Image, edited_image: Image.Image,
                 instruction: str, **kwargs) -> Dict[str, Any]:
        """
        Evaluate local attribute accuracy using LMM VQA-2Level.
        
        Returns:
            Dictionary with 'score' (0-1) and 'details'
        """
        user_prompt = Prompts.VQA_2LEVEL_USER.format(prompt=instruction)
        
        probs = self.lmm.get_logits_for_choices(
            images=[original_image, edited_image],
            system_prompt=Prompts.VQA_2LEVEL_SYSTEM,
            user_prompt=user_prompt,
            choices=["Yes", "No"]
        )
        
        score = probs.get("Yes", 0.0)
        
        return {
            "score": score,
            "details": {
                "method": "LMM VQA-2Level",
                "probabilities": probs,
                "instruction": instruction
            }
        }


class ActionStateChangeAccuracyEvaluator(BaseEvaluator):
    """
    Dimension 10: Action/State Change Accuracy
    Pipeline: LMM VQA-2Level
    
    Evaluates the faithful execution of action or state changes.
    """
    
    def __init__(self, config: EvaluatorConfig):
        super().__init__(config)
        self.lmm = LMMHandler(config)
    
    @property
    def dimension_name(self) -> str:
        return "Action_State_Change_Accuracy"
    
    def evaluate(self, original_image: Image.Image, edited_image: Image.Image,
                 instruction: str, **kwargs) -> Dict[str, Any]:
        """
        Evaluate action/state change accuracy using LMM VQA-2Level.
        
        Returns:
            Dictionary with 'score' (0-1) and 'details'
        """
        user_prompt = Prompts.VQA_2LEVEL_USER.format(prompt=instruction)
        
        probs = self.lmm.get_logits_for_choices(
            images=[original_image, edited_image],
            system_prompt=Prompts.VQA_2LEVEL_SYSTEM,
            user_prompt=user_prompt,
            choices=["Yes", "No"]
        )
        
        score = probs.get("Yes", 0.0)
        
        return {
            "score": score,
            "details": {
                "method": "LMM VQA-2Level",
                "probabilities": probs,
                "instruction": instruction
            }
        }


class SpatialAccuracyEvaluator(BaseEvaluator):
    """
    Dimension 11: Spatial Accuracy
    Pipeline: LMM VQA-2Level
    
    Evaluates the faithful execution of spatial descriptions (e.g., "to the left of").
    """
    
    def __init__(self, config: EvaluatorConfig):
        super().__init__(config)
        self.lmm = LMMHandler(config)
    
    @property
    def dimension_name(self) -> str:
        return "Spatial_Accuracy"
    
    def evaluate(self, original_image: Image.Image, edited_image: Image.Image,
                 instruction: str, **kwargs) -> Dict[str, Any]:
        """
        Evaluate spatial accuracy using LMM VQA-2Level.
        
        Returns:
            Dictionary with 'score' (0-1) and 'details'
        """
        user_prompt = Prompts.VQA_2LEVEL_USER.format(prompt=instruction)
        
        probs = self.lmm.get_logits_for_choices(
            images=[original_image, edited_image],
            system_prompt=Prompts.VQA_2LEVEL_SYSTEM,
            user_prompt=user_prompt,
            choices=["Yes", "No"]
        )
        
        score = probs.get("Yes", 0.0)
        
        return {
            "score": score,
            "details": {
                "method": "LMM VQA-2Level",
                "probabilities": probs,
                "instruction": instruction
            }
        }


class TextContentStyleAccuracyEvaluator(BaseEvaluator):
    """
    Dimension 12: Text Content & Style Accuracy
    Pipeline: Multi-step Hybrid (GOT-OCR2.0 + LMM VQA-5Level)
    
    Assesses the model's ability to render text with correct spelling and appropriate style.
    Uses Eq. (5) from the paper for score fusion.
    """
    
    def __init__(self, config: EvaluatorConfig):
        super().__init__(config)
        self.got_ocr = GOTOCRHandler(config)
        self.lmm = LMMHandler(config)
    
    @property
    def dimension_name(self) -> str:
        return "Text_Content_Style_Accuracy"
    
    def evaluate(self, original_image: Image.Image, edited_image: Image.Image,
                 instruction: str, target_text: str = None, **kwargs) -> Dict[str, Any]:
        """
        Evaluate text content and style accuracy using Hybrid OCR + LMM pipeline.
        
        Process:
        1. GOT-OCR extracts text and computes content accuracy (Sc)
        2. LMM evaluates style/position (Ss via VQA-5Level, 1-5 score)
        3. Scores are fused using Eq. (5)
        
        Args:
            target_text: The expected text to be rendered (extracted from instruction if not provided)
        
        Returns:
            Dictionary with 'score' (0-1) and 'details'
        """
        # Extract target text from instruction if not provided
        if target_text is None:
            target_text = self._extract_target_text(instruction)
        
        # Step 1: Content Accuracy using GOT-OCR
        extracted_text = self.got_ocr.extract_text(edited_image)
        sc = self.got_ocr.compute_content_accuracy(extracted_text, target_text)
        
        # Step 2: Style/Position using LMM VQA-5Level
        style_prompt = Prompts.TEXT_STYLE_PROMPT + f"""

[User Prompt]: {instruction}
[Target Text]: {target_text}

Please evaluate the style and position compliance of the text in the image."""
        
        style_response = self.lmm.generate(
            images=[edited_image],
            system_prompt="You are an expert evaluator for text rendering.",
            user_prompt=style_prompt,
            max_tokens=512
        )
        
        # Parse style score from JSON response
        ss = self._parse_style_score(style_response)
        
        # Step 3: Score fusion using Eq. (5)
        # Ss is normalized from 1-5 to 0-1
        ss_normalized = (ss - 1) / 4
        
        # Content accuracy penalty factor
        if sc >= 1.0:
            content_factor = 1.0
        elif sc >= 0.8:
            content_factor = 0.8
        elif sc >= 0.6:
            content_factor = 0.5
        else:
            content_factor = 0.1
        
        final_score = ss_normalized * content_factor
        
        return {
            "score": final_score,
            "details": {
                "method": "Hybrid GOT-OCR + LMM VQA-5Level",
                "content_accuracy_sc": sc,
                "style_score_ss": ss,
                "style_score_normalized": ss_normalized,
                "content_factor": content_factor,
                "extracted_text": extracted_text,
                "target_text": target_text,
                "style_response": style_response
            }
        }
    
    def _extract_target_text(self, instruction: str) -> str:
        """Extract target text from instruction (text in quotes)."""
        # Look for text in quotes
        matches = re.findall(r'"([^"]+)"|\'([^\']+)\'', instruction)
        if matches:
            return matches[0][0] or matches[0][1]
        
        # Look for common patterns like "Write X" or "Add text X"
        write_match = re.search(r'(?:write|add|render|put|display)\s+(?:the\s+)?(?:text\s+)?["\']?([^"\'.,]+)["\']?', 
                               instruction, re.IGNORECASE)
        if write_match:
            return write_match.group(1).strip()
        
        return ""
    
    def _parse_style_score(self, response: str) -> float:
        """Parse style score from LMM response."""
        try:
            # Try to extract JSON
            json_match = re.search(r'\{[^{}]*"score"\s*:\s*(\d+)[^{}]*\}', response, re.DOTALL)
            if json_match:
                return float(json_match.group(1))
            
            # Try to find score directly
            score_match = re.search(r'score["\']?\s*:\s*(\d+)', response, re.IGNORECASE)
            if score_match:
                return float(score_match.group(1))
            
        except (ValueError, AttributeError):
            pass
        
        return 3.0  # Default to middle score


class WorldKnowledgeReasoningEvaluator(BaseEvaluator):
    """
    Dimension 13: World Knowledge & Reasoning
    Pipeline: LMM VQA-2Level
    
    Evaluates instructions requiring external knowledge (e.g., "add a statue of [famous person]").
    """
    
    def __init__(self, config: EvaluatorConfig):
        super().__init__(config)
        self.lmm = LMMHandler(config)
    
    @property
    def dimension_name(self) -> str:
        return "World_Knowledge_Reasoning"
    
    def evaluate(self, original_image: Image.Image, edited_image: Image.Image,
                 instruction: str, hint: str = None, **kwargs) -> Dict[str, Any]:
        """
        Evaluate world knowledge & reasoning using LMM VQA-2Level.
        
        Args:
            hint: Evaluation criteria/hint for what to check (optional)
        
        Returns:
            Dictionary with 'score' (0-1) and 'details'
        """
        if hint is None:
            hint = self._generate_hint(instruction)
        
        user_prompt = Prompts.WORLD_KNOWLEDGE_USER.format(
            prompt=instruction,
            hint=hint
        )
        
        probs = self.lmm.get_logits_for_choices(
            images=[original_image, edited_image],
            system_prompt=Prompts.WORLD_KNOWLEDGE_SYSTEM,
            user_prompt=user_prompt,
            choices=["Yes", "No"]
        )
        
        score = probs.get("Yes", 0.0)
        
        return {
            "score": score,
            "details": {
                "method": "LMM VQA-2Level",
                "probabilities": probs,
                "instruction": instruction,
                "hint": hint
            }
        }
    
    def _generate_hint(self, instruction: str) -> str:
        """Generate evaluation hint from instruction."""
        return f"Check if the edit correctly implements the knowledge-based requirement: {instruction}"


class SubjectIdentityFidelityEvaluator(BaseEvaluator):
    """
    Dimension 14: Subject Identity Fidelity
    Pipeline: Hybrid LMM-Specialist (Sa2VA + DINOv3)
    
    Assesses whether a subject's core identity is preserved during attribute modifications.
    """
    
    def __init__(self, config: EvaluatorConfig):
        super().__init__(config)
        # Don't initialize handlers here - we'll do it on-demand with memory management
        self.config = config
    
    @property
    def dimension_name(self) -> str:
        return "Subject_Identity_Fidelity"
    
    def evaluate(self, original_image: Image.Image, edited_image: Image.Image,
                 instruction: str, **kwargs) -> Dict[str, Any]:
        """
        Evaluate subject identity fidelity using Hybrid LMM-Specialist pipeline.
        
        Unlike Non-Edited Region Fidelity, this directly compares subject regions
        (without mask inversion).
        
        Note: This method carefully manages GPU memory by loading/unloading
        models sequentially since LMM (~16GB) + Sa2VA (~14GB) > 24GB GPU.
        
        Returns:
            Dictionary with 'score' (0-1) and 'details'
        """
        # Step 1: Generate segmentation commands with LMM
        seg_prompt = Prompts.SUBJECT_IDENTITY_SEGMENTATION + f"""

Editing Instruction: {instruction}

Output only the JSON list."""
        
        lmm = LMMHandler(self.config)
        seg_response = lmm.generate(
            images=[original_image, edited_image],
            system_prompt="You are an expert in image editing analysis.",
            user_prompt=seg_prompt,
            max_tokens=256
        )
        
        # Parse segmentation commands
        seg_commands = self._parse_seg_commands(seg_response)
        
        if not seg_commands:
            seg_commands = ["Please segment the main subject"]
        
        # Clear LMM from GPU before loading Sa2VA
        clear_gpu_memory()
        
        # Step 2: Segment with Sa2VA
        sa2va = Sa2VAHandler(self.config)
        
        # Store all masks before clearing Sa2VA
        masks = []
        for cmd in seg_commands:
            mask_orig = sa2va.segment(original_image, cmd)
            mask_edit = sa2va.segment(edited_image, cmd)
            masks.append((cmd, mask_orig, mask_edit))
        
        # Clear Sa2VA from GPU before loading DINOv3
        clear_gpu_memory()
        
        # Step 3: Compute similarity with DINOv3
        dino = DINOv3Handler(self.config)
        
        # Get image sizes for mask resizing
        orig_size = (original_image.width, original_image.height)
        edit_size = (edited_image.width, edited_image.height)
        
        similarities = []
        seg_details = []
        
        for cmd, mask_orig, mask_edit in masks:
            # Resize masks to match respective image sizes if needed
            if mask_orig.shape[:2] != (original_image.height, original_image.width):
                mask_orig_resized = np.array(Image.fromarray((mask_orig * 255).astype(np.uint8)).resize(
                    orig_size, Image.Resampling.NEAREST)) / 255.0
            else:
                mask_orig_resized = mask_orig
                
            if mask_edit.shape[:2] != (edited_image.height, edited_image.width):
                mask_edit_resized = np.array(Image.fromarray((mask_edit * 255).astype(np.uint8)).resize(
                    edit_size, Image.Resampling.NEAREST)) / 255.0
            else:
                mask_edit_resized = mask_edit
            
            features_orig = dino.extract_features(original_image, mask_orig_resized)
            features_edit = dino.extract_features(edited_image, mask_edit_resized)
            
            sim = dino.compute_similarity(features_orig, features_edit)
            similarities.append(sim)
            
            seg_details.append({
                "command": cmd,
                "similarity": sim
            })
        
        # Average similarity
        score = np.mean(similarities) if similarities else 0.0
        
        return {
            "score": score,
            "details": {
                "method": "Hybrid LMM-Specialist (Sa2VA + DINOv3)",
                "segmentation_details": seg_details,
                "num_subjects": len(seg_commands)
            }
        }
    
    def _parse_seg_commands(self, response: str) -> List[str]:
        """Parse segmentation commands from response."""
        try:
            # Try to find JSON list
            json_match = re.search(r'\[([^\]]+)\]', response)
            if json_match:
                items = re.findall(r'"([^"]+)"', json_match.group())
                return [item for item in items if item.startswith("Please segment")]
        except:
            pass
        
        # Find patterns directly
        commands = re.findall(r'Please segment [^"]+', response)
        return commands


class CompositionInteractionEvaluator(BaseEvaluator):
    """
    Dimension 15: Composition & Interaction
    Pipeline: LMM Multi-Question VQA
    
    Evaluates the logical interaction and visual coherence between 
    newly introduced/altered elements and existing scene content.
    """
    
    def __init__(self, config: EvaluatorConfig):
        super().__init__(config)
        self.lmm = LMMHandler(config)
    
    @property
    def dimension_name(self) -> str:
        return "Composition_Interaction"
    
    def evaluate(self, original_image: Image.Image, edited_image: Image.Image,
                 instruction: str, **kwargs) -> Dict[str, Any]:
        """
        Evaluate composition & interaction using LMM Multi-Question VQA.
        
        Returns:
            Dictionary with 'score' (0-1) and 'details'
        """
        # Stage 1: Question Generation
        qg_prompt = Prompts.COMPOSITION_INTERACTION_QG.format(prompt=instruction)
        
        questions_response = self.lmm.generate(
            images=[original_image],
            system_prompt="You are an expert in visual arts and composition evaluation.",
            user_prompt=qg_prompt,
            max_tokens=1024
        )
        
        # Parse questions
        questions = self._parse_questions(questions_response)
        
        if not questions:
            questions = [
                "Is the edited element placed in a logical position within the scene?",
                "Is the scale of the edited element consistent with other objects?",
                "Does the perspective of the edited element match the scene?",
                "Are the interactions between elements visually believable?",
                "Is the occlusion (in front of/behind) of elements correct?"
            ]
        
        # Stage 2: Question Answering
        yes_count = 0
        qa_results = []
        
        for q in questions:
            qa_prompt = Prompts.PHYSICAL_PLAUSIBILITY_QA.format(question=q)
            
            answer = self.lmm.generate(
                images=[original_image, edited_image],
                system_prompt="You are an expert evaluator. Answer Yes or No only.",
                user_prompt=qa_prompt,
                max_tokens=10
            )
            
            is_yes = "yes" in answer.lower()
            if is_yes:
                yes_count += 1
            
            qa_results.append({
                "question": q,
                "answer": answer,
                "is_yes": is_yes
            })
        
        score = yes_count / len(questions) if questions else 0.0
        
        return {
            "score": score,
            "details": {
                "method": "LMM Multi-Question VQA",
                "num_questions": len(questions),
                "yes_count": yes_count,
                "qa_results": qa_results
            }
        }
    
    def _parse_questions(self, response: str) -> List[str]:
        """Parse questions from JSON response."""
        try:
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data.get("composition_questions", [])
        except json.JSONDecodeError:
            pass
        
        questions = re.findall(r'(?:Question \d+:?\s*)?([^\n?]+\?)', response)
        return questions[:7] if questions else []


# ==============================================================================
# MAIN EVALUATOR CLASS - Aggregates all 15 dimensions
# ==============================================================================

class I2IBenchSingleReferenceEvaluator:
    """
    Main evaluator class that aggregates all 15 single-reference evaluation dimensions.
    
    Usage:
        config = EvaluatorConfig()
        evaluator = I2IBenchSingleReferenceEvaluator(config)
        results = evaluator.evaluate_all(original_image, edited_image, instruction)
    """
    
    # Dimension names mapping
    DIMENSION_NAMES = {
        "overall_image_quality": "Overall_Image_Quality",
        "aesthetic_quality": "Aesthetic_Quality",
        "blending_naturalness": "Blending_Naturalness",
        "editing_artifacts": "Editing_Artifacts",
        "instruction_following_macro": "Instruction_Following_Macro",
        "non_edited_region_fidelity": "Non_Edited_Region_Fidelity",
        "physical_plausibility": "Physical_Plausibility",
        "object_manipulation_accuracy": "Object_Manipulation_Accuracy",
        "local_attribute_accuracy": "Local_Attribute_Accuracy",
        "action_state_change_accuracy": "Action_State_Change_Accuracy",
        "spatial_accuracy": "Spatial_Accuracy",
        "text_content_style_accuracy": "Text_Content_Style_Accuracy",
        "world_knowledge_reasoning": "World_Knowledge_Reasoning",
        "subject_identity_fidelity": "Subject_Identity_Fidelity",
        "composition_interaction": "Composition_Interaction"
    }
    
    def __init__(self, config: Optional[EvaluatorConfig] = None):
        """
        Initialize the evaluator with all 15 dimensions.
        
        Args:
            config: EvaluatorConfig instance (uses defaults if not provided)
        """
        self.config = config or EvaluatorConfig()
        
        # Initialize all evaluators
        self.evaluators = {
            # Common Dimensions (7)
            "overall_image_quality": OverallImageQualityEvaluator(self.config),
            "aesthetic_quality": AestheticQualityEvaluator(self.config),
            "blending_naturalness": BlendingNaturalnessEvaluator(self.config),
            "editing_artifacts": EditingArtifactsEvaluator(self.config),
            "instruction_following_macro": InstructionFollowingMacroEvaluator(self.config),
            "non_edited_region_fidelity": NonEditedRegionFidelityEvaluator(self.config),
            "physical_plausibility": PhysicalPlausibilityEvaluator(self.config),
            
            # Specific Dimensions (8)
            "object_manipulation_accuracy": ObjectManipulationAccuracyEvaluator(self.config),
            "local_attribute_accuracy": LocalAttributeAccuracyEvaluator(self.config),
            "action_state_change_accuracy": ActionStateChangeAccuracyEvaluator(self.config),
            "spatial_accuracy": SpatialAccuracyEvaluator(self.config),
            "text_content_style_accuracy": TextContentStyleAccuracyEvaluator(self.config),
            "world_knowledge_reasoning": WorldKnowledgeReasoningEvaluator(self.config),
            "subject_identity_fidelity": SubjectIdentityFidelityEvaluator(self.config),
            "composition_interaction": CompositionInteractionEvaluator(self.config)
        }
    
    def evaluate_dimension(self, dimension: str, original_image: Image.Image,
                          edited_image: Image.Image, instruction: str,
                          **kwargs) -> Dict[str, Any]:
        """
        Evaluate a single dimension.
        
        Args:
            dimension: Name of the dimension to evaluate
            original_image: Original input image
            edited_image: Edited output image
            instruction: Editing instruction
            **kwargs: Additional arguments for specific dimensions
            
        Returns:
            Dictionary with 'score' and 'details'
        """
        if dimension not in self.evaluators:
            raise ValueError(f"Unknown dimension: {dimension}. "
                           f"Available: {list(self.evaluators.keys())}")
        
        return self.evaluators[dimension].evaluate(
            original_image, edited_image, instruction, **kwargs
        )
    
    def evaluate_all(self, original_image: Image.Image, edited_image: Image.Image,
                    instruction: str, dimensions: Optional[List[str]] = None,
                    **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all (or selected) dimensions.
        
        Args:
            original_image: Original input image
            edited_image: Edited output image
            instruction: Editing instruction
            dimensions: List of dimensions to evaluate (all if None)
            **kwargs: Additional arguments for specific dimensions
            
        Returns:
            Dictionary mapping dimension names to their results
        """
        if dimensions is None:
            dimensions = list(self.evaluators.keys())
        
        results = {}
        for dim in dimensions:
            try:
                print(f"Evaluating dimension: {dim}...")
                results[dim] = self.evaluate_dimension(
                    dim, original_image, edited_image, instruction, **kwargs
                )
            except Exception as e:
                print(f"Error evaluating {dim}: {e}")
                results[dim] = {
                    "score": 0.0,
                    "details": {"error": str(e)}
                }
            
            # Memory management: clear GPU memory after each dimension
            # This is necessary because different dimensions use different models
            # and the RTX 3090 Ti (24GB) can't hold all models simultaneously
            clear_gpu_memory()
        
        return results
    
    def get_summary_scores(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Extract just the scores from evaluation results.
        
        Returns:
            Dictionary mapping dimension names to scores
        """
        return {dim: res.get("score", 0.0) for dim, res in results.items()}
    
    def compute_overall_score(self, results: Dict[str, Dict[str, Any]]) -> float:
        """
        Compute an overall score by averaging all dimension scores.
        
        Returns:
            Average score across all evaluated dimensions
        """
        scores = self.get_summary_scores(results)
        if not scores:
            return 0.0
        return sum(scores.values()) / len(scores)
    
    @staticmethod
    def load_image(path: str) -> Image.Image:
        """Load an image from file path."""
        return Image.open(path).convert("RGB")
    
    @staticmethod
    def list_dimensions() -> List[str]:
        """List all available evaluation dimensions."""
        return [
            "overall_image_quality",
            "aesthetic_quality", 
            "blending_naturalness",
            "editing_artifacts",
            "instruction_following_macro",
            "non_edited_region_fidelity",
            "physical_plausibility",
            "object_manipulation_accuracy",
            "local_attribute_accuracy",
            "action_state_change_accuracy",
            "spatial_accuracy",
            "text_content_style_accuracy",
            "world_knowledge_reasoning",
            "subject_identity_fidelity",
            "composition_interaction"
        ]


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def find_image_file(directory: str, prefix: str) -> Optional[str]:
    """
    Find an image file in directory with given prefix.
    Supports .png, .jpg, .jpeg extensions.
    """
    for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
        path = os.path.join(directory, prefix + ext)
        if os.path.exists(path):
            return path
    return None


def save_results(results: Dict[str, Any], output_path: str):
    """Save evaluation results to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def load_results(input_path: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(input_path, 'r') as f:
        return json.load(f)


# ==============================================================================
# MAIN ENTRY POINT (for standalone testing)
# ==============================================================================

if __name__ == "__main__":
    print("I2I-Bench Single-Reference Evaluator Module")
    print("=" * 50)
    print("Available Evaluation Dimensions:")
    for i, dim in enumerate(I2IBenchSingleReferenceEvaluator.list_dimensions(), 1):
        print(f"  {i}. {dim}")
    print("\nUse the runner script to evaluate test cases.")
