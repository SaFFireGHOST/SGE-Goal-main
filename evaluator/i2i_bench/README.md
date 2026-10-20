# I2I-Bench Single-Reference Evaluator

An implementation of the 15 single-reference evaluation dimensions from the I2I-Bench paper:
> **"I2I-Bench: A Comprehensive Benchmark Suite for Image-to-Image Editing Models"**
> Wang et al., arXiv:2512.04660v1

## Overview

This implementation provides automated evaluation of language-guided image editing (LGIE) results using a combination of specialist models and Large Multimodal Models (LMMs), exactly as described in the I2I-Bench paper.

### Evaluation Dimensions

The 15 single-reference dimensions are divided into:

#### Common Dimensions (7) - Applicable to all SE tasks

| # | Dimension | Method | Tool(s) |
|---|-----------|--------|---------|
| 1 | **Overall Image Quality** | Specialist | Q-Insight |
| 2 | **Aesthetic Quality** | Specialist | ArtiMuse |
| 3 | **Blending Naturalness** | LMM VQA-5Level | Qwen3-VL-8B-Instruct |
| 4 | **Editing Artifacts** | LMM VQA-5Level | Qwen3-VL-8B-Instruct |
| 5 | **Instruction Following (Macro)** | LMM VQA-5Level | Qwen3-VL-8B-Instruct |
| 6 | **Non-Edited Region Fidelity** | Hybrid LMM-Specialist | Sa2VA + DINOv3 |
| 7 | **Physical Plausibility** | LMM Multi-Question VQA | Qwen3-VL-8B-Instruct |

#### Specific Dimensions (8) - For specific instruction types

| # | Dimension | Method | Tool(s) |
|---|-----------|--------|---------|
| 8 | **Object Manipulation Accuracy** | LMM VQA-2Level | Qwen3-VL-8B-Instruct |
| 9 | **Local Attribute Accuracy** | LMM VQA-2Level | Qwen3-VL-8B-Instruct |
| 10 | **Action/State Change Accuracy** | LMM VQA-2Level | Qwen3-VL-8B-Instruct |
| 11 | **Spatial Accuracy** | LMM VQA-2Level | Qwen3-VL-8B-Instruct |
| 12 | **Text Content & Style Accuracy** | Hybrid OCR + LMM | GOT-OCR2.0 + LMM |
| 13 | **World Knowledge & Reasoning** | LMM VQA-2Level | Qwen3-VL-8B-Instruct |
| 14 | **Subject Identity Fidelity** | Hybrid LMM-Specialist | Sa2VA + DINOv3 |
| 15 | **Composition & Interaction** | LMM Multi-Question VQA | Qwen3-VL-8B-Instruct |

## Evaluation Methods

### 1. Specialist Models
Direct quantitative scores from specialized pre-trained models:
- **Q-Insight**: Technical image quality assessment
- **ArtiMuse**: Aesthetic/artistic quality assessment

### 2. LMM VQA-5Level (Equation 1)
5-level rating scale: excellent (1.0), good (0.75), fair (0.5), poor (0.25), bad (0.0)

```
Score = Σ(weight_i × P(choice_i))
```

### 3. LMM VQA-2Level (Equation 4)
Binary Yes/No judgment:
```
Score = P('Yes')
```

### 4. LMM Multi-Question VQA (Equation 3)
Two-stage process:
1. Question Generation: LMM generates N Yes/No questions
2. Question Answering: LMM answers each question
```
Score = (1/N) × Σ I(answer_i = 'Yes')
```

### 5. Hybrid LMM-Specialist (Equation 2)
Combines LMM semantic understanding with specialist precision:
1. LMM generates segmentation command
2. Sa2VA performs segmentation
3. DINOv3 extracts features
4. Cosine similarity computed
```
Score = (f_orig · f_gen) / (||f_orig|| × ||f_gen||)
```

### 6. Hybrid OCR + LMM (Equation 5)
For text rendering evaluation:
1. GOT-OCR2.0 extracts text and computes content accuracy (Sc)
2. LMM evaluates style/position (Ss)
3. Scores fused with piecewise function

## Installation

```bash
# Clone or navigate to the directory
cd /path/to/I2I-Bench

# Install dependencies
pip install -r requirements.txt

# For GPU support (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Data Format

Organize your test cases in the `data/` directory:

```
data/
├── 1/
│   ├── original_image.png (or .jpg/.jpeg)
│   ├── edited_image.png (or .jpg/.jpeg)
│   └── edit_instruction.txt
├── 2/
│   ├── original_image.jpg
│   ├── edited_image.jpg
│   └── edit_instruction.txt
├── 3/
│   └── ...
└── ...
```

Each test case directory should contain:
- `original_image.png/jpg/jpeg`: The source/reference image
- `edited_image.png/jpg/jpeg`: The edited/generated image
- `edit_instruction.txt`: The editing instruction/prompt (text file)

## Usage

### Basic Usage

```bash
# Run all dimensions on all test cases
python runner.py

# Run on specific test cases
python runner.py --test-cases 1 2 3

# Run specific dimensions
python runner.py --dimensions blending_naturalness physical_plausibility

# Use custom data directory
python runner.py --data-dir /path/to/custom/data
```

### List Available Options

```bash
# List all available dimensions
python runner.py --list-dimensions

# List discovered test cases
python runner.py --list-test-cases
```

### Programmatic Usage

```python
from i2i_bench_evaluator import I2IBenchSingleReferenceEvaluator, EvaluatorConfig
from PIL import Image

# Initialize evaluator
config = EvaluatorConfig()
evaluator = I2IBenchSingleReferenceEvaluator(config)

# Load images
original = Image.open("path/to/original.png")
edited = Image.open("path/to/edited.png")
instruction = "Change the color of the car to red"

# Evaluate single dimension
result = evaluator.evaluate_dimension(
    "blending_naturalness",
    original, edited, instruction
)
print(f"Score: {result['score']:.4f}")

# Evaluate all dimensions
all_results = evaluator.evaluate_all(original, edited, instruction)
for dim, res in all_results.items():
    print(f"{dim}: {res['score']:.4f}")
```

## Output Structure

Results are saved in each test case directory:

```
data/
├── 1/
│   ├── original_image.png
│   ├── edited_image.png
│   ├── edit_instruction.txt
│   └── evaluation_results/
│       ├── blending_naturalness/
│       │   ├── result.json      # Full result with details
│       │   ├── score.txt        # Just the score
│       │   └── summary.txt      # Human-readable summary
│       ├── physical_plausibility/
│       │   └── ...
│       └── ...
├── aggregate_report/
│   ├── aggregate_report_YYYYMMDD_HHMMSS.json
│   ├── aggregate_report_YYYYMMDD_HHMMSS.txt
│   └── scores_YYYYMMDD_HHMMSS.csv
└── ...
```

### Result JSON Format

```json
{
  "dimension": "blending_naturalness",
  "instruction": "Change the color of the vase to yellow",
  "score": 0.75,
  "details": {
    "method": "LMM VQA-5Level",
    "probabilities": {
      "excellent": 0.0,
      "good": 1.0,
      "fair": 0.0,
      "poor": 0.0,
      "bad": 0.0
    },
    "response": "good"
  },
  "timestamp": "2026-01-29T12:00:00"
}
```

## Configuration

Customize the evaluator through `EvaluatorConfig`:

```python
from i2i_bench_evaluator import EvaluatorConfig

config = EvaluatorConfig(
    # LMM Model (Generalist)
    lmm_model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    
    # Specialist Models
    q_insight_model_name="q-future/Q-Insight",
    artimuse_model_name="artimuse/ArtiMuse",
    got_ocr_model_name="stepfun-ai/GOT-OCR2.0",
    sa2va_model_name="ByteDance/Sa2VA-4B",
    dinov3_model_name="facebook/dinov2-large",
    
    # Device
    device="cuda",  # or "cpu"
    
    # VQA-5Level weights
    vqa_5level_weights={
        "excellent": 1.0,
        "good": 0.75,
        "fair": 0.5,
        "poor": 0.25,
        "bad": 0.0
    }
)
```

## Notes

### Model Substitutions
Due to model availability, some substitutions are made:
- **Qwen3-VL-8B-Instruct** → Qwen2.5-VL-7B-Instruct
- **DINOv3** → DINOv2-Large

### Fallback Behavior
If specialist models are unavailable, the evaluators fall back to:
- Basic image statistics for quality metrics
- Simple feature extraction for similarity metrics
- Mock responses for testing

### Performance
- First run will download required models (~10-20GB)
- GPU is highly recommended for reasonable inference times
- Batch processing is done sequentially to manage memory

## Citation

If you use this implementation, please cite the original I2I-Bench paper:

```bibtex
@article{wang2025i2ibench,
  title={I2I-Bench: A Comprehensive Benchmark Suite for Image-to-Image Editing Models},
  author={Wang, Juntong and Wang, Jiarui and Duan, Huiyu and Kang, Jiaxiang and Zhai, Guangtao and Min, Xiongkuo},
  journal={arXiv preprint arXiv:2512.04660},
  year={2025}
}
```

## License

This implementation is for research purposes. Please refer to the original I2I-Bench paper and the licenses of individual models used.
