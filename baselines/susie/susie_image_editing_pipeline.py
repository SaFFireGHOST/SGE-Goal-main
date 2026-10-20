#!/usr/bin/env python3
"""
SuSIE Image Editing Pipeline for Final Subgoal Generation

This script implements SuSIE's image-editing pipeline for subgoal generation.
Based on the paper "Zero-Shot Robotic Manipulation with Pretrained Image-Editing
Diffusion Models" by Black et al.

The pipeline uses a finetuned InstructPix2Pix model to iteratively generate
subgoal images. Each iteration produces an intermediate subgoal that represents
progress toward completing the manipulation task specified by the text instruction.

Key parameters from the paper (Table 5):
- ktest = 20: subgoal sampling interval during execution
- kmax = 14: maximum steps ahead for subgoal generation during training
- The model generates subgoals ~kmax steps into the future

For typical manipulation tasks, 3-5 iterations are usually sufficient to reach
the final goal state, as evidenced by Figure 8 (CALVIN rollouts) in the paper.

Usage:
    nohup bash -c 'python3 susie_image_editing_pipeline.py --image_path original_image_1.jpg --instruction "Put the drumstick on the plate" --output_path final_subgoal.jpg --num_iterations 5 --save_intermediate --use_gpu' > inference.log 2>&1 &

Source: Built based on SuSIE paper and codebase (kvablack/susie)
"""

import argparse
import os
import sys
from pathlib import Path


def configure_jax_backend(use_gpu: bool):
    """
    Configure JAX backend BEFORE importing JAX or any JAX-dependent modules.
    This must be called before any other imports that use JAX.
    """
    if not use_gpu:
        # Force JAX to use CPU only - must be set before JAX is imported
        os.environ['JAX_PLATFORMS'] = 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    # If use_gpu is True, let JAX auto-detect GPU


def _delayed_imports():
    """Import JAX-dependent modules after backend is configured."""
    global np, Image, create_sample_fn, initialize_compilation_cache
    
    import numpy as np
    from PIL import Image
    
    # Add the susie package to path
    sys.path.insert(0, str(Path(__file__).parent / "susie"))
    
    from susie.model import create_sample_fn
    from susie.jax_utils import initialize_compilation_cache
    
    return np, Image, create_sample_fn, initialize_compilation_cache


def load_and_preprocess_image(image_path: str, target_size: tuple = (256, 256)) -> tuple:
    """
    Load and preprocess an image for the SuSIE pipeline.
    
    The SuSIE model was trained on 256x256 images and REQUIRES this resolution.
    From the paper (Section 4.3): "Our diffusion model and policy operate on 
    images of size 256 × 256."
    
    Args:
        image_path: Path to the input image
        target_size: Target size (width, height). Must be (256, 256) for SuSIE.
    
    Returns:
        Tuple of (preprocessed image as numpy array (256, 256, 3), original_size tuple)
    """
    import numpy as np
    from PIL import Image
    
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (width, height)
    
    # SuSIE model requires exactly 256x256 images
    if image.size != target_size:
        image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    return np.array(image), original_size


def save_image(image, output_path: str):
    """
    Save a numpy array as an image file.
    
    Args:
        image: Image array of shape (H, W, 3), dtype uint8
        output_path: Path to save the image
    """
    from PIL import Image as PILImage
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    PILImage.fromarray(image).save(output_path)
    print(f"Saved image to: {output_path}")


def run_susie_pipeline(
    image_path: str,
    instruction: str,
    output_path: str,
    num_iterations: int = 5,
    model_path: str = "kvablack/susie",
    num_timesteps: int = 50,
    prompt_w: float = 7.5,
    context_w: float = 2.5,
    eta: float = 0.0,
    save_intermediate: bool = True,
    pretrained_path: str = "runwayml/stable-diffusion-v1-5",
    use_gpu: bool = False,
):
    """
    Run the SuSIE image editing pipeline to generate the final subgoal image.
    
    According to the SuSIE paper (Algorithm 1), the system iteratively generates
    subgoals by:
    1. Generating a subgoal image given current observation and language command
    2. The low-level policy executes actions to reach that subgoal (in simulation/real robot)
    3. After ktest steps, regenerate a new subgoal from the current state
    4. Repeat until task completion
    
    For image-only generation (without a robot), we simulate this by iteratively
    feeding the generated subgoal back as the input for the next iteration.
    This produces a chain of subgoal images representing progress toward the final goal.
    
    NOTE: In the actual SuSIE robotic system, each subgoal is generated from the
    REAL robot observation, not from the previous generated subgoal. This script
    approximates that behavior for visualization purposes.
    
    From the paper:
    - kmax (training) = 14 for BridgeData, 22 for CALVIN
    - ktest (evaluation) = 20 for both real-world and CALVIN
    - Figure 8 shows typical tasks require 3-5 subgoal generations
    - Image size: 256x256 (required by the model, Section 4.3)
    - DDIM sampler with 50 steps (Appendix A.1.1)
    - Image guidance (context_w): 2.5, Text guidance (prompt_w): 7.5
    
    Args:
        image_path: Path to the input/original scene image
        instruction: Text instruction describing the manipulation task
                    (e.g., "Put the drumstick on the plate")
        output_path: Path to save the final output image
        num_iterations: Number of times to run the subgoal generation.
                       Recommended: 3-5 for typical manipulation tasks.
                       From paper Figure 8: most CALVIN tasks complete in 3-5 iterations.
        model_path: Path to the SuSIE UNet weights (HuggingFace or local)
        num_timesteps: Number of DDIM sampling timesteps (default: 50, from paper)
        prompt_w: Text guidance weight for classifier-free guidance (default: 7.5, from paper)
        context_w: Image context guidance weight (default: 2.5, from paper)
        eta: DDIM eta parameter (0.0 = deterministic DDIM, from paper)
        save_intermediate: Whether to save intermediate subgoal images (default: True)
        pretrained_path: Path to pretrained Stable Diffusion for VAE/text encoder
        use_gpu: Whether to use GPU for inference (default: False, uses CPU).
                 GPU mode provides faster inference but requires sufficient VRAM.
    
    Returns:
        Final subgoal image as numpy array of shape (256, 256, 3), dtype uint8
    """
    # Import required modules (JAX backend already configured by main)
    sys.path.insert(0, str(Path(__file__).parent / "susie"))
    from susie.model import create_sample_fn
    from susie.jax_utils import initialize_compilation_cache
    
    print("=" * 60)
    print("SuSIE Image Editing Pipeline for Subgoal Generation")
    print("=" * 60)
    print(f"\nInput image: {image_path}")
    print(f"Instruction: \"{instruction}\"")
    print(f"Number of iterations: {num_iterations}")
    print(f"Diffusion timesteps: {num_timesteps}")
    print(f"Text guidance weight: {prompt_w}")
    print(f"Image guidance weight: {context_w}")
    print(f"Execution mode: {'GPU' if use_gpu else 'CPU'}")
    print()
    
    # Import JAX and check device availability
    import jax
    devices = jax.devices()
    gpu_devices = [d for d in devices if d.platform == 'gpu']
    
    if use_gpu:
        if gpu_devices:
            print(f"GPU mode enabled. Found {len(gpu_devices)} GPU(s): {gpu_devices}")
        else:
            print("WARNING: GPU mode requested but no GPU found. Check JAX GPU installation.")
    else:
        print(f"CPU mode enabled. Using CPU for inference (slower but memory-efficient).")
        print(f"Available devices: {devices}")
    
    # Initialize JAX compilation cache for faster subsequent runs
    print("Initializing JAX compilation cache...")
    initialize_compilation_cache()
    
    # Create the sampling function
    print(f"Loading SuSIE model from: {model_path}")
    print("(This may take a moment on first run as weights are downloaded and compiled...)")
    sample_fn = create_sample_fn(
        path=model_path,
        num_timesteps=num_timesteps,
        prompt_w=prompt_w,
        context_w=context_w,
        eta=eta,
        pretrained_path=pretrained_path,
    )
    print("Model loaded successfully!\n")
    
    # Load the original image
    # NOTE: SuSIE model requires exactly 256x256 images (paper Section 4.3)
    print("Loading and preprocessing input image...")
    current_image, original_size = load_and_preprocess_image(image_path)
    print(f"Original image size: {original_size[0]}x{original_size[1]}")
    print(f"Processing at resolution: 256x256 (required by SuSIE model)")
    
    if save_intermediate:
        # Save the original image as iteration 0
        base_path = Path(output_path)
        intermediate_path = base_path.parent / f"{base_path.stem}_iter_0{base_path.suffix}"
        save_image(current_image, str(intermediate_path))
    
    # Iteratively generate subgoals
    print(f"\nRunning {num_iterations} iterations of subgoal generation...")
    print("-" * 40)
    
    for iteration in range(1, num_iterations + 1):
        print(f"\nIteration {iteration}/{num_iterations}:")
        print(f"  Generating subgoal for: \"{instruction}\"")
        
        # Generate the next subgoal
        # The sample_fn expects:
        # - image: numpy array of shape (256, 256, 3), dtype uint8
        # - prompt: string describing the edit/task
        generated_image = sample_fn(current_image, instruction)
        
        print(f"  Subgoal generated (shape: {generated_image.shape})")
        
        if save_intermediate:
            intermediate_path = base_path.parent / f"{base_path.stem}_iter_{iteration}{base_path.suffix}"
            save_image(generated_image, str(intermediate_path))
            print(f"  Saved intermediate: {intermediate_path}")
        
        # Update current image for next iteration
        current_image = generated_image
    
    print("-" * 40)
    print(f"\nPipeline complete! Generated {num_iterations} subgoal(s).")
    
    # Save the final output
    save_image(current_image, output_path)
    print(f"\nFinal output saved to: {output_path}")
    
    return current_image


def main():
    parser = argparse.ArgumentParser(
        description="SuSIE Image Editing Pipeline for Final Subgoal Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings (5 iterations)
  python susie_image_editing_pipeline.py \\
      --image_path original_image_1.jpg \\
      --instruction "Put the drumstick on the plate" \\
      --output_path final_subgoal.jpg

  # With custom number of iterations
  python susie_image_editing_pipeline.py \\
      --image_path original_image_1.jpg \\
      --instruction "Put the drumstick on the plate" \\
      --output_path final_subgoal.jpg \\
      --num_iterations 3

  # Save all intermediate subgoals
  python susie_image_editing_pipeline.py \\
      --image_path original_image_1.jpg \\
      --instruction "Put the drumstick on the plate" \\
      --output_path final_subgoal.jpg \\
      --save_intermediate

Notes:
  - The model weights are automatically downloaded from HuggingFace on first run
  - Recommended iterations: 3-5 for typical manipulation tasks (from paper experiments)
  - Uses classifier-free guidance with default weights from the paper:
    * Text guidance (prompt_w): 7.5
    * Image guidance (context_w): 2.5
        """
    )
    
    parser.add_argument(
        "--image_path", 
        type=str, 
        required=True,
        help="Path to the input/original scene image"
    )
    parser.add_argument(
        "--instruction", 
        type=str, 
        required=True,
        help='Text instruction for the manipulation task (e.g., "Put the drumstick on the plate")'
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        required=True,
        help="Path to save the final subgoal image"
    )
    parser.add_argument(
        "--num_iterations", 
        type=int, 
        default=5,
        help="Number of subgoal generation iterations (default: 5, recommended: 3-5)"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="kvablack/susie",
        help="Path to SuSIE model weights (HuggingFace or local path)"
    )
    parser.add_argument(
        "--num_timesteps", 
        type=int, 
        default=50,
        help="Number of diffusion sampling timesteps (default: 50)"
    )
    parser.add_argument(
        "--prompt_w", 
        type=float, 
        default=7.5,
        help="Text guidance weight for classifier-free guidance (default: 7.5)"
    )
    parser.add_argument(
        "--context_w", 
        type=float, 
        default=2.5,
        help="Image context guidance weight for classifier-free guidance (default: 2.5)"
    )
    parser.add_argument(
        "--eta", 
        type=float, 
        default=0.0,
        help="DDIM eta parameter: 0.0=deterministic, 1.0=stochastic (default: 0.0)"
    )
    parser.add_argument(
        "--save_intermediate",
        action="store_true",
        default=True,
        help="Save intermediate subgoal images from each iteration (default: True)"
    )
    parser.add_argument(
        "--no_save_intermediate",
        action="store_true",
        help="Disable saving intermediate subgoal images"
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to pretrained Stable Diffusion for VAE/text encoder (use :revision suffix for specific revision)"
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use GPU for inference (faster but requires sufficient VRAM). Default is CPU mode."
    )
    
    args = parser.parse_args()
    
    # Configure JAX backend BEFORE importing JAX-dependent modules
    configure_jax_backend(args.use_gpu)
    
    # Now import JAX-dependent modules
    _delayed_imports()
    
    # Validate input image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Input image not found: {args.image_path}")
        sys.exit(1)
    
    # Handle save_intermediate flag
    save_intermediate = args.save_intermediate and not args.no_save_intermediate
    
    # Run the pipeline
    run_susie_pipeline(
        image_path=args.image_path,
        instruction=args.instruction,
        output_path=args.output_path,
        num_iterations=args.num_iterations,
        model_path=args.model_path,
        num_timesteps=args.num_timesteps,
        prompt_w=args.prompt_w,
        context_w=args.context_w,
        eta=args.eta,
        save_intermediate=save_intermediate,
        pretrained_path=args.pretrained_path,
        use_gpu=args.use_gpu,
    )


if __name__ == "__main__":
    main()
