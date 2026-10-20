# LGIE Evaluation - Modal Runner

This directory contains a Modal wrapper script to run the LGIE evaluation scripts on Modal's serverless GPU infrastructure.

## Prerequisites

1. **Install Modal CLI:**
   ```bash
   pip install modal
   modal token new  # Authenticate with Modal
   ```

2. **Create HuggingFace Secret (for model downloads):**
   ```bash
   modal secret create huggingface-secret HF_TOKEN=hf_your_token_here
   ```

3. **Ensure model weights are available locally:**
   - `sam_vit_h_4b8939.pth` (~2.4 GB) - SAM checkpoint
   - `XMem/saves/XMem.pth` - XMem model weights
   - `loftr_weights/*.ckpt` - LoFTR checkpoints

## Quick Start

```bash
# First time: Upload large model weights to Modal volume
modal run modal_eval_runner.py --action upload

# Verify weights are uploaded
modal run modal_eval_runner.py --action check_weights

# Run an evaluation script
modal run modal_eval_runner.py --script R1.py
modal run modal_eval_runner.py --script R2.py
modal run modal_eval_runner.py --script R3.py
# ... etc.
```

## Usage

### List Available Scripts
```bash
modal run modal_eval_runner.py --action list
```

Output:
```
Available evaluation scripts:
  - R1.py
  - R2.py
  - R3.py
  - R4.py
  - R5-6-7.py
  - R8-9.py
  - R10-11-12.py
  - R13.py
  - R14-15.py
```

### Run Evaluation Scripts
```bash
# Basic run
modal run modal_eval_runner.py --script R1.py

# With verbose output
modal run modal_eval_runner.py --script R2.py --verbose

# Pass extra arguments to the script
modal run modal_eval_runner.py --script R1.py --extra-args "--some-flag value"
```

### Manage Model Weights
```bash
# Upload weights to Modal volume (run once)
modal run modal_eval_runner.py --action upload

# Check what weights are available
modal run modal_eval_runner.py --action check_weights
```

### View Outputs
```bash
# List output files in Modal volume
modal run modal_eval_runner.py --action list_outputs
```

## Configuration

### GPU Settings
Edit the following variables in `modal_eval_runner.py` to change GPU configuration:

```python
GPU_TYPE = "a100"  # Options: "t4", "a10g", "a100", "a100-80gb", "h100"
GPU_MEMORY = "40GB"
```

### Timeout Settings
Default timeout is 24 hours. For longer runs, modify the `timeout` parameter in the function decorators.

## How It Works

1. **Image Build:** Modal builds a Docker image with all required Python dependencies (PyTorch, transformers, SAM, etc.)

2. **Weight Management:** Large model weights (SAM ~2.4GB, XMem, LoFTR) are stored in a Modal Volume to avoid re-uploading each run

3. **Script Execution:** The evaluation script runs on a GPU instance with:
   - All dependencies pre-installed
   - Weights symlinked from the volume to expected paths
   - `PYTHONPATH` configured for XMem/LoFTR imports
   - Output streamed to your terminal in real-time

4. **Output Persistence:** Results are saved to a Modal Volume and can be downloaded

## Evaluation Scripts

| Script | Description |
|--------|-------------|
| `R1.py` | Instruction-target objects that remained stationary |
| `R2.py` | Unmentioned moved-within-scene objects |
| `R3.py` | Unmentioned removed-from-scene objects |
| `R4.py` | Hallucinated objects detection |
| `R5-6-7.py` | Object change detection (color, texture, shape) |
| `R8-9.py` | Lighting changes detection (brighter/darker) |
| `R10-11-12.py` | Camera transforms (zoom, perspective) |
| `R13.py` | Photorealism loss detection |
| `R14-15.py` | Instruction adherence (source & destination) |

## Dependencies Installed

The Modal image includes:
- PyTorch 2.4.0 with CUDA 12.4
- transformers 4.40.1
- segment-anything (SAM)
- XMem dependencies
- LoFTR dependencies
- opencv-python, pillow, imageio, av, decord
- scikit-learn, scikit-image, scipy
- flash-attn (optional)
- And more...

## Troubleshooting

### "SAM checkpoint not found"
Run `modal run modal_eval_runner.py --action upload` to upload weights.

### "Out of GPU memory"
Try changing `GPU_TYPE` to `"a100-80gb"` or `"h100"`.

### "Module not found: XMem"
The PYTHONPATH should be automatically configured. Check that XMem directory exists in the project.

### Slow image builds
The first build downloads all dependencies and may take 10-15 minutes. Subsequent runs use cached images.

## Cost Estimation

Modal pricing (approximate, check modal.com for current rates):
- A100 40GB: ~$3.00/hour
- A100 80GB: ~$4.50/hour
- H100: ~$5.50/hour

Most evaluation scripts complete in 10-60 minutes depending on test data size.
