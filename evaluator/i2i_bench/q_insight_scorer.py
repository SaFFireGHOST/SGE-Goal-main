#!/usr/bin/env python
"""
Q-Insight Scorer - Standalone script for image quality scoring using Q-Align/OneAlign
This script is designed to run in a separate environment with transformers 4.36.1
"""

import sys
import json
import base64
import torch
from io import BytesIO
from PIL import Image

def get_quality_score(image_data: str) -> dict:
    """
    Score image quality using Q-Insight (OneAlign) model.
    
    Args:
        image_data: Base64-encoded image data
        
    Returns:
        dict with 'score' key containing quality score (1-5 scale)
    """
    from transformers import AutoModelForCausalLM
    
    # Decode image
    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    
    # Load model (cached after first load)
    model = AutoModelForCausalLM.from_pretrained(
        "q-future/one-align", 
        trust_remote_code=True, 
        attn_implementation="eager",
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    # Get quality score
    score = model.score([image], task_="quality", input_="image")
    
    # Convert to float
    if hasattr(score, 'item'):
        score = score.item()
    elif isinstance(score, (list, tuple)):
        score = float(score[0])
    else:
        score = float(score)
    
    return {"score": score}


def main():
    """Main entry point - reads base64 image from stdin, outputs JSON."""
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Test mode - create a random image
        import numpy as np
        test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        buffer = BytesIO()
        test_image.save(buffer, format="PNG")
        image_data = base64.b64encode(buffer.getvalue()).decode()
    else:
        # Read base64-encoded image from stdin
        image_data = sys.stdin.read().strip()
    
    try:
        result = get_quality_score(image_data)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
