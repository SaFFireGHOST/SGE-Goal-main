#!/usr/bin/env python
"""
CLOVER Visual Planner with Chained Planning and MiDaS Depth Estimation

This script implements an extended Visual Planner based on CLOVER (Closed-Loop 
Visuomotor Control with Generative Expectation for Robotic Manipulation).

================================================================================
IMPLEMENTATION APPROACH
================================================================================

1. BASE VISUAL PLANNER (from CLOVER)
------------------------------------
   The core visual planner is a text-conditioned video diffusion model that 
   generates K=8 future sub-goal frames from:
   - A current RGB-D observation (condition image)
   - A natural language task instruction (e.g., "open the drawer")
   
   Model Architecture:
   - UNet backbone with causal temporal attention (Imagen-based)
   - CLIP text encoder (ViT-Large-Patch14) for instruction embedding
   - V-parameterization objective with cosine noise schedule
   - DDIM sampling with 20 steps for efficient inference
   - Classifier-free guidance (weight=4.0) for better instruction following
   
   The model outputs RGB-D frames (4 channels: RGB + depth), generating 
   consistent visual plans that respect both appearance and 3D geometry.

2. CHAINED PLANNING STRATEGY (with Early Stopping)
--------------------------------------------------
   Following CLOVER's Algorithm 1, we implement iterative replanning:
   
   - Each planning cycle generates K=8 sub-goals
   - When sub-goals are exhausted, replan from the last generated frame
   - Chain multiple cycles to handle complex, long-horizon tasks
   
   Key Innovation - Avoiding Depth Drift:
   For subsequent chains, we use the model's OWN generated depth output as 
   the depth condition for the next chain, rather than re-estimating with 
   MiDaS. This is critical because:
   - MiDaS on 128x128 generated images produces noisy/inconsistent depth
   - The model was trained to generate coherent RGB-D pairs
   - Using generated depth maintains temporal consistency across chains
   
   Early Stopping (Convergence-Based):
   Since we lack an external task oracle, we implement a convergence heuristic:
   - Compute cosine distance between consecutive sub-goals
   - If min_distance < threshold (0.001), task appears complete
   - This detects when the model generates static/converged frames
   
   Replanning on Failure (DR Threshold):
   - If max frame distance > DR=1.0, plan is deemed unreachable
   - Regenerate the plan (up to 5 attempts per chain)

3. MiDaS DEPTH ESTIMATION
-------------------------
   For the INITIAL input image only, we use MiDaS DPT_Large for monocular 
   depth estimation:
   
   Why MiDaS:
   - Provides accurate relative depth from a single RGB image
   - DPT_Large offers high-quality dense predictions
   - Enables using the visual planner on arbitrary real-world images
   
   Depth Convention Matching:
   - MiDaS outputs INVERSE depth (closer = higher values)
   - CALVIN dataset uses DISTANCE (closer = lower values)
   - We INVERT MiDaS output: depth = -midas_prediction
   - Then min-max normalize to [-1, 1] to match training distribution
   
   Important: MiDaS is ONLY used for the initial condition image. For 
   chained planning, we reuse the model's generated depth to avoid 
   quality degradation on small generated images.

================================================================================
HYPERPARAMETERS (from CLOVER Paper)
================================================================================
   - Image size: 128x128
   - Sub-goals per chain (K): 8
   - Diffusion steps: 100 (training) / 20 (DDIM sampling)
   - Beta schedule: cosine
   - Objective: pred_v (v-parameterization)
   - Text guidance weight: 4.0
   - Replan distance threshold (DR): 1.0
   - Convergence threshold: 0.001 (for early stopping)
   - Max chains: 10 (configurable, for complex tasks)

================================================================================
REFERENCES
================================================================================
   Paper: "Closed-Loop Visuomotor Control with Generative Expectation 
          for Robotic Manipulation" (NeurIPS 2024)
   Code:  https://github.com/OpenDriveLab/CLOVER
   MiDaS: https://github.com/isl-org/MiDaS

================================================================================
USAGE
================================================================================
   Basic:
     python clover_visual_planner.py --image input.jpg --instruction "open drawer"
   
   With chaining and annotations:
     python clover_visual_planner.py --image input.jpg --instruction "open drawer" \\
         --output_dir output --num_chains 5 --annotate_chains
   
   All options:
     --image             Path to input RGB image
     --instruction       Natural language task instruction
     --checkpoint        Path to visual planner weights (default: checkpoints/visual_planner.pt)
     --output_dir        Directory for outputs (frames, video, GIF)
     --num_chains        Max planning chains (default: 10)
     --convergence_threshold  Early stop threshold (default: 0.001)
     --annotate_chains   Add chain number labels to output frames
     --no_auto_stop      Disable early stopping, always use all chains
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional
import imageio
from tqdm import tqdm
import cv2

# Add CLOVER paths to sys.path
CLOVER_ROOT = Path(__file__).parent / "CLOVER"
VISUAL_PLANNER_ROOT = CLOVER_ROOT / "visual_planner"
sys.path.insert(0, str(VISUAL_PLANNER_ROOT))
sys.path.insert(0, str(CLOVER_ROOT))

from visual_planner import VisualPlanner
from trainer import GoalGaussianDiffusion, normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from transformers import CLIPTextModel, CLIPTokenizer


class CLOVERVisualPlannerInference:
    """
    CLOVER Visual Planner for generating visual plans from text instructions.
    
    This class implements the Visual Planner component as described in CLOVER:
    - Uses a text-conditioned video diffusion model (based on Imagen architecture)
    - Generates K=8 sub-goals (future frames) as visual plans
    - Includes replanning strategy for unreachable sub-goals (DR=1.0 threshold)
    - Uses DDIM sampling with 20 steps for efficiency
    - Text guidance weight of 4.0 for better instruction following
    
    Hyperparameters from CLOVER paper (Section 3.1 and Appendix B):
    - Image size: 128x128
    - K (number of sub-goals): 8
    - Diffusion steps: 100 (training) / 20 (DDIM sampling)
    - Beta schedule: cosine
    - Objective: pred_v (v-parameterization)
    - Text guidance weight: 4.0
    - Distance threshold for replan (DR): 1.0
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        image_size: Tuple[int, int] = (128, 128),
        num_frames: int = 8,  # K=8 sub-goals as per paper
        diffusion_steps: int = 100,
        sampling_steps: int = 20,  # DDIM sampling steps
        guidance_weight: float = 4.0,  # Text guidance weight from paper
        with_depth: bool = True,  # Model was trained with depth
        replan_threshold: float = 1.0,  # DR threshold from paper
    ):
        """
        Initialize the CLOVER Visual Planner.
        
        Args:
            checkpoint_path: Path to the visual planner checkpoint
            device: Device to run inference on
            image_size: Input/output image size (128x128 as per paper)
            num_frames: Number of future frames to generate (K=8)
            diffusion_steps: Total diffusion timesteps
            sampling_steps: DDIM sampling steps (20 for efficiency)
            guidance_weight: Classifier-free guidance weight (4.0)
            with_depth: Whether model includes depth prediction (uses MiDaS)
            replan_threshold: Distance threshold for replanning (DR=1.0)
        """
        self.device = device
        self.image_size = image_size
        self.num_frames = num_frames
        self.guidance_weight = guidance_weight
        self.with_depth = with_depth
        self.replan_threshold = replan_threshold
        
        # Initialize MiDaS depth estimation model (required for depth)
        self.midas_model = None
        self.midas_transform = None
        if with_depth:
            print("Loading MiDaS depth estimation model (DPT_Large)...")
            # Load MiDaS model - using DPT_Large for best quality
            self.midas_model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
            self.midas_model.to(device)
            self.midas_model.eval()
            
            # Load MiDaS transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.midas_transform = midas_transforms.dpt_transform
            print("  MiDaS DPT_Large model loaded successfully!")
        
        print(f"Initializing CLOVER Visual Planner...")
        print(f"  - Image size: {image_size}")
        print(f"  - Number of sub-goals (K): {num_frames}")
        print(f"  - Sampling steps: {sampling_steps}")
        print(f"  - Guidance weight: {guidance_weight}")
        print(f"  - Replan threshold (DR): {replan_threshold}")
        
        # Initialize CLIP text encoder for conditioning
        print("Loading CLIP text encoder...")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder.to(device)
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)
        
        # Initialize visual planner model
        # Model architecture from paper: UNet with causal temporal attention
        # in_channels: 6 for RGB-only (3 RGB + 3 conditioned), 8 for RGB-D
        # out_channels: 3 for RGB-only, 4 for RGB-D
        in_channels = 8 if with_depth else 6
        out_channels = 4 if with_depth else 3
        
        print("Initializing Visual Planner model...")
        model = VisualPlanner(
            image_size=image_size[0],
            in_channels=in_channels,
            out_channels=out_channels,
            dims=3,  # 3D convolutions for temporal
            temporal_length=num_frames,
            use_vae=False,
            decoupled_output=False,
            decoupled_input=False,
            flow_reg=False,  # Disable during inference
            with_state_estimate=False,
        )
        
        # Wrap model with diffusion
        print("Initializing diffusion model...")
        self.diffusion = GoalGaussianDiffusion(
            model=model,
            image_size=image_size,
            channels=3,
            timesteps=diffusion_steps,
            sampling_timesteps=sampling_steps,
            loss_type='l2',
            objective='pred_v',  # V-parameterization as per paper
            beta_schedule='cosine',  # Cosine schedule from paper
            min_snr_loss_weight=True,
            auto_normalize=False,
            with_depth=with_depth,
            use_vae=False,
        )
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Prefer EMA weights for inference (more stable/better quality)
        # The checkpoint structure is:
        # - 'model': regular model weights
        # - 'ema': EMA wrapper state dict containing 'ema_model.*' keys
        if 'ema' in checkpoint:
            ema_dict = checkpoint['ema']
            # Extract ema_model weights and remove the 'ema_model.' prefix
            state_dict = {k.replace('ema_model.', ''): v 
                         for k, v in ema_dict.items() 
                         if k.startswith('ema_model.')}
            print(f"  Loaded EMA model weights ({len(state_dict)} parameters)")
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
            print(f"  Loaded regular model weights ({len(state_dict)} parameters)")
        else:
            state_dict = checkpoint
            print(f"  Loaded raw checkpoint ({len(state_dict)} parameters)")
            
        # Load state dict
        missing, unexpected = self.diffusion.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Warning: Missing keys: {len(missing)} (e.g., {missing[:3]}...)")
        if unexpected:
            print(f"  Warning: Unexpected keys: {len(unexpected)} (e.g., {unexpected[:3]}...)")
        
        self.diffusion.to(device)
        self.diffusion.eval()
        
        print("Visual Planner initialized successfully!")
        
    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text instruction using CLIP text encoder.
        
        Args:
            text: Natural language instruction (e.g., "Put the drumstick on the plate")
            
        Returns:
            Text embedding tensor of shape (1, seq_len, 768)
        """
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=32
        ).to(self.device)
        
        with torch.no_grad():
            text_embed = self.text_encoder(**inputs).last_hidden_state
            
        return text_embed
    
    def preprocess_image(self, image_path: str, depth_path: Optional[str] = None) -> torch.Tensor:
        """
        Preprocess input image for the visual planner.
        
        The model expects RGB-D input where:
        - RGB is normalized: (x / 255 - 0.5) / 0.5 → [-1, 1]
        - Depth is min-max normalized per image, then scaled to [-1, 1]
        
        Depth is estimated using MiDaS DPT_Large model, which provides accurate
        monocular depth estimation. The MiDaS output (inverse depth) is inverted
        to match CALVIN's depth convention (distance: closer = lower values).
        
        Args:
            image_path: Path to the RGB input image
            depth_path: Optional path to depth map (if None, MiDaS is used)
            
        Returns:
            Preprocessed image tensor of shape (1, 4, H, W) if with_depth else (1, 3, H, W)
            (Note: No frame dimension - x_cond in CLOVER is (B, C, H, W))
        """
        # Load original image (keep full resolution for MiDaS)
        image_original = Image.open(image_path).convert('RGB')
        
        # Resize for model input
        image = image_original.resize(self.image_size, Image.BILINEAR)
        
        # Convert to tensor and normalize to [0, 1]
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        
        # Rearrange from (H, W, C) to (C, H, W)
        image_tensor = image_tensor.permute(2, 0, 1)
        
        # Apply normalization: (x - 0.5) / 0.5 = x * 2 - 1 → [-1, 1]
        # This matches CALVIN dataset preprocessing: Normalize(mean=[0.5]*3, std=[0.5]*3)
        image_tensor = image_tensor * 2 - 1
        
        if self.with_depth:
            if depth_path is not None:
                # Use provided depth map
                print("  Using provided depth map...")
                depth_img = Image.open(depth_path)
                depth_img = depth_img.resize(self.image_size, Image.BILINEAR)
                depth_tensor = torch.from_numpy(np.array(depth_img)).float()
                
                # Ensure single channel
                if depth_tensor.dim() == 3:
                    depth_tensor = depth_tensor.mean(dim=2)  # Average if multi-channel
                depth_tensor = depth_tensor.unsqueeze(0)  # (1, H, W)
            else:
                # Use MiDaS for depth estimation
                print("  Estimating depth using MiDaS DPT_Large...")
                depth_tensor = self._estimate_depth_midas(image_original)
                
            # Min-max normalize depth to [0, 1], then scale to [-1, 1]
            # This matches CALVIN: (depth - min) / (max - min), then * 2 - 1
            depth_min = depth_tensor.min()
            depth_max = depth_tensor.max()
            if depth_max - depth_min > 1e-6:
                depth_tensor = (depth_tensor - depth_min) / (depth_max - depth_min)
            else:
                depth_tensor = depth_tensor * 0 + 0.5  # Constant depth if no variation
            depth_tensor = depth_tensor * 2 - 1  # Scale to [-1, 1]
            
            # Concatenate RGB + depth: (4, H, W)
            image_tensor = torch.cat([image_tensor, depth_tensor], dim=0)
        
        # Add batch dimension only: (1, C, H, W)
        # Note: CLOVER's x_cond is (B, C, H, W), the model internally repeats it across frames
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    @torch.no_grad()
    def _estimate_depth_midas(self, image: Image.Image) -> torch.Tensor:
        """
        Estimate depth using MiDaS model.
        
        MiDaS provides monocular depth estimation that captures true 3D geometry,
        which is crucial for CLOVER to understand spatial relationships between
        the robot arm and objects.
        
        IMPORTANT: MiDaS outputs INVERSE depth (closer = higher values), but
        CALVIN uses DISTANCE (closer = lower values). We invert the MiDaS output
        to match CALVIN's convention.
        
        Args:
            image: PIL Image (original resolution, will be processed by MiDaS)
            
        Returns:
            Depth tensor of shape (1, H, W) resized to self.image_size
        """
        # Convert PIL image to numpy array for MiDaS
        img_np = np.array(image)
        
        # Apply MiDaS transform
        input_batch = self.midas_transform(img_np).to(self.device)
        
        # Run MiDaS inference
        prediction = self.midas_model(input_batch)
        
        # Resize to target size (MiDaS outputs at different resolution)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=self.image_size,
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
        # MiDaS outputs INVERSE depth (closer = higher values)
        # CALVIN uses DISTANCE (closer = lower values, far = higher values)
        # We need to INVERT MiDaS output to match CALVIN convention
        # Inversion: depth = max - midas_output (or simply negate before normalization)
        depth_tensor = -prediction.cpu()  # Invert: now far = higher values
        
        # Ensure (1, H, W) shape
        if depth_tensor.dim() == 2:
            depth_tensor = depth_tensor.unsqueeze(0)
        
        return depth_tensor
    
    def compute_frame_distances(self, frames: torch.Tensor) -> List[float]:
        """
        Compute cosine distances between consecutive generated frames.
        
        This is used for the replanning strategy: if consecutive frames
        have distance > DR (1.0), the plan is considered unreachable.
        
        Args:
            frames: Generated frames tensor of shape (1, K, C, H, W)
            
        Returns:
            List of distances between consecutive frames
        """
        distances = []
        frames_flat = frames.squeeze(0)  # (K, C, H, W)
        
        for i in range(len(frames_flat) - 1):
            frame_i = frames_flat[i].flatten()
            frame_j = frames_flat[i + 1].flatten()
            
            # Compute cosine distance
            cos_sim = F.cosine_similarity(frame_i.unsqueeze(0), frame_j.unsqueeze(0))
            cos_dist = 1 - cos_sim.item()
            distances.append(cos_dist)
            
        return distances
    
    def is_plan_reachable(self, frames: torch.Tensor) -> Tuple[bool, float]:
        """
        Check if the generated plan is reachable based on frame consistency.
        
        From the CLOVER paper (Section 3.3 and Algorithm 1):
        - Measure cosine distance between consecutive generated sub-goals
        - If max distance > DR (1.0), the plan is unreachable
        
        Args:
            frames: Generated frames tensor
            
        Returns:
            Tuple of (is_reachable, max_distance)
        """
        distances = self.compute_frame_distances(frames)
        max_dist = max(distances) if distances else 0
        
        return max_dist <= self.replan_threshold, max_dist
    
    @torch.no_grad()
    def generate_visual_plan(
        self,
        condition_image: torch.Tensor,
        text_instruction: str,
        max_replan_attempts: int = 5,
    ) -> Tuple[torch.Tensor, bool, int]:
        """
        Generate a visual plan (sequence of sub-goals) for the given instruction.
        
        Implements the Visual Planner from CLOVER (Section 3.1):
        1. Encode text instruction using CLIP
        2. Generate K future frames using video diffusion
        3. Check plan reachability
        4. Replan if necessary
        
        Args:
            condition_image: Initial observation tensor (1, C, H, W) already in [-1, 1] range
            text_instruction: Natural language task description
            max_replan_attempts: Maximum replanning attempts
            
        Returns:
            Tuple of (generated_frames, success, num_attempts)
        """
        # Encode text
        text_embed = self.encode_text(text_instruction)
        
        # x_cond is already normalized to [-1, 1] from preprocess_image
        # No additional normalization needed
        x_cond = condition_image
        
        for attempt in range(max_replan_attempts):
            print(f"  Generating plan (attempt {attempt + 1}/{max_replan_attempts})...")
            
            # Sample from diffusion model using DDIM
            generated = self.diffusion.sample(
                x_cond=x_cond,
                task_embed=text_embed,
                batch_size=1,
                frames=self.num_frames,
                guidance_weight=self.guidance_weight,
                return_all_timesteps=False,
            )
            
            # generated shape: (1, K, C, H, W) in [-1, 1] range
            # The model outputs are in [-1, 1] normalized space
            # With auto_normalize=False, the unnormalize is identity in the model,
            # so we need to manually convert to [0, 1] range here
            
            # Unnormalize from [-1, 1] to [0, 1]
            generated = unnormalize_to_zero_to_one(generated)
            
            # Store the full RGBD output for potential use in chaining
            # This avoids re-estimating depth with MiDaS on generated images
            if self.with_depth:
                # Convert back to [-1, 1] for storage (consistent format)
                self._last_generated_rgbd = generated * 2 - 1  # (1, K, 4, H, W) in [-1, 1]
                rgb_frames = generated[:, :, :3, :, :]
            else:
                self._last_generated_rgbd = None
                rgb_frames = generated
            
            # Clamp to valid [0, 1] range (safety measure for any outliers)
            rgb_frames = torch.clamp(rgb_frames, 0, 1)
            
            # Check if plan is reachable
            is_reachable, max_dist = self.is_plan_reachable(rgb_frames)
            
            print(f"    Max frame distance: {max_dist:.4f} (threshold: {self.replan_threshold})")
            
            if is_reachable:
                print(f"  Plan is reachable! Generated {self.num_frames} sub-goals.")
                return rgb_frames, True, attempt + 1
            else:
                print(f"    Plan unreachable (distance > threshold). Replanning...")
        
        print(f"  Warning: Failed to generate reachable plan after {max_replan_attempts} attempts.")
        print(f"  Using last generated plan anyway.")
        return rgb_frames, False, max_replan_attempts
    
    def _compute_frame_convergence(self, frames: torch.Tensor) -> Tuple[float, float]:
        """
        Compute how much the frames are converging (becoming similar).
        
        This is used for early stopping in chained planning:
        - If consecutive frames become very similar, the task may be complete
        - Based on the CLOVER paper's error measurement using cosine distance
        
        Args:
            frames: Generated frames tensor (1, K, C, H, W) in [0, 1] range
            
        Returns:
            Tuple of (mean_distance, min_distance) between consecutive frames
        """
        frames_flat = frames.squeeze(0)  # (K, C, H, W)
        K = frames_flat.shape[0]
        
        if K < 2:
            return 1.0, 1.0
        
        distances = []
        for i in range(K - 1):
            frame_i = frames_flat[i].flatten()
            frame_j = frames_flat[i + 1].flatten()
            
            # Cosine distance (1 - cosine_similarity)
            cos_sim = F.cosine_similarity(frame_i.unsqueeze(0), frame_j.unsqueeze(0))
            cos_dist = 1 - cos_sim.item()
            distances.append(cos_dist)
        
        return np.mean(distances), np.min(distances)
    
    @torch.no_grad()
    def generate_chained_visual_plan(
        self,
        condition_image: torch.Tensor,
        text_instruction: str,
        num_chains: int = 2,
        max_replan_attempts: int = 5,
        convergence_threshold: float = 0.005,
        auto_stop: bool = True,
    ) -> Tuple[torch.Tensor, bool, int, List[int]]:
        """
        Generate an extended visual plan by chaining multiple planning cycles.
        
        Following the CLOVER paper's Algorithm 1 (Test-time Execution):
        - The visual planner generates K=8 sub-goals per cycle
        - When all K sub-goals would be exhausted, replan from current state
        - Loop continues until time limit OR task completion
        
        Since we don't have a task oracle to detect completion, we use a
        convergence-based heuristic: if consecutive sub-goals become very 
        similar (low cosine distance), the task appears to be complete.
        
        Args:
            condition_image: Initial observation tensor (1, C, H, W) in [-1, 1] range
            text_instruction: Natural language task description
            num_chains: Maximum number of planning cycles (total frames ≤ 8 * num_chains)
            max_replan_attempts: Maximum replanning attempts per chain (DR threshold)
            convergence_threshold: Stop early if min frame distance falls below this
            auto_stop: If True, stop when frames converge (task appears complete)
            
        Returns:
            Tuple of (generated_frames, success, total_attempts, chain_labels)
            chain_labels is a list of chain numbers (1-indexed) for each frame
        """
        all_frames = []
        all_chain_labels = []
        current_condition = condition_image
        total_attempts = 0
        all_success = True
        stopped_early = False
        
        for chain_idx in range(num_chains):
            print(f"\n  Chain {chain_idx + 1}/{num_chains}:")
            
            # Generate visual plan for this chain
            frames, success, attempts = self.generate_visual_plan(
                condition_image=current_condition,
                text_instruction=text_instruction,
                max_replan_attempts=max_replan_attempts,
            )
            
            total_attempts += attempts
            if not success:
                all_success = False
            
            # Check for convergence (task completion heuristic)
            # Following CLOVER's error measurement philosophy:
            # If consecutive sub-goals become very similar, the task appears complete
            if auto_stop and chain_idx > 0:
                mean_dist, min_dist = self._compute_frame_convergence(frames)
                print(f"    Frame convergence: mean_dist={mean_dist:.4f}, min_dist={min_dist:.4f}")
                
                if min_dist < convergence_threshold:
                    print(f"    Task appears complete (min_dist < {convergence_threshold}). Stopping early.")
                    stopped_early = True
                    # Still include these frames but don't continue
                    if chain_idx == 0:
                        all_frames.append(frames)
                        all_chain_labels.extend([chain_idx + 1] * frames.shape[1])
                    else:
                        all_frames.append(frames[:, 1:, :, :, :])
                        all_chain_labels.extend([chain_idx + 1] * (frames.shape[1] - 1))
                    break
            
            # Collect frames (skip first frame if not the first chain to avoid duplicates)
            if chain_idx == 0:
                all_frames.append(frames)
                all_chain_labels.extend([chain_idx + 1] * frames.shape[1])
            else:
                # Skip the first frame as it should be similar to the last frame of previous chain
                all_frames.append(frames[:, 1:, :, :, :])
                all_chain_labels.extend([chain_idx + 1] * (frames.shape[1] - 1))
            
            # Check if this is the last chain - don't need to prepare next condition
            if chain_idx == num_chains - 1:
                break
                
            # Use the last generated frame as the new condition for the next chain
            # This follows CLOVER's Algorithm 1: when sub-goals are exhausted, replan from current state
            #
            # IMPORTANT: We use the model's own generated RGB-D output, not re-estimated depth.
            # Re-estimating depth with MiDaS on small generated images causes drift/hallucinations.
            # The model was trained to generate consistent RGB-D pairs, so we trust its depth.
            
            # Get the last frame from the full generated output (not just RGB)
            # Need to access the full RGBD output from generate_visual_plan
            # For now, we'll store the full generated output and retrieve it here
            last_frame_rgb = frames[:, -1, :, :, :]  # (1, 3, H, W) in [0, 1]
            last_frame_rgb = last_frame_rgb * 2 - 1  # Convert to [-1, 1]
            
            # Need to add depth channel for the new condition
            if self.with_depth:
                # Use the model's generated depth from the last frame instead of MiDaS
                # This avoids drift from re-estimating depth on generated images
                if hasattr(self, '_last_generated_rgbd') and self._last_generated_rgbd is not None:
                    # Use the model's own generated depth (already in [-1, 1])
                    last_frame_depth = self._last_generated_rgbd[:, -1, 3:, :, :]  # (1, 1, H, W)
                    print(f"    Using model's generated depth for next chain (avoiding MiDaS drift)")
                else:
                    # Fallback to MiDaS if generated depth not available
                    print(f"    Warning: Using MiDaS depth (may cause drift)")
                    last_frame_np = ((last_frame_rgb.squeeze(0).cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
                    last_frame_np = last_frame_np.transpose(1, 2, 0)  # (H, W, 3)
                    last_frame_pil = Image.fromarray(last_frame_np)
                    
                    depth_tensor = self._estimate_depth_midas(last_frame_pil)
                    depth_min = depth_tensor.min()
                    depth_max = depth_tensor.max()
                    if depth_max - depth_min > 1e-6:
                        depth_tensor = (depth_tensor - depth_min) / (depth_max - depth_min)
                    else:
                        depth_tensor = depth_tensor * 0 + 0.5
                    last_frame_depth = (depth_tensor * 2 - 1).unsqueeze(0).unsqueeze(0).to(self.device)
                
                # Concatenate RGB + depth
                current_condition = torch.cat([last_frame_rgb, last_frame_depth], dim=1)
            else:
                current_condition = last_frame_rgb
        
        # Concatenate all frames
        all_frames = torch.cat(all_frames, dim=1)
        
        chains_used = chain_idx + 1
        if stopped_early:
            print(f"\n  Chained planning complete (early stop): {all_frames.shape[1]} total frames from {chains_used}/{num_chains} chains")
        else:
            print(f"\n  Chained planning complete: {all_frames.shape[1]} total frames from {chains_used} chains")
        
        return all_frames, all_success, total_attempts, all_chain_labels
    
    def save_frames(
        self,
        frames: torch.Tensor,
        output_dir: str,
        prefix: str = "subgoal",
        chain_labels: Optional[List[int]] = None,
        annotate_chains: bool = False,
    ) -> List[str]:
        """
        Save individual sub-goal frames as images.
        
        Args:
            frames: Generated frames tensor (1, K, C, H, W)
            output_dir: Directory to save frames
            prefix: Filename prefix
            chain_labels: Optional list of chain numbers for each frame
            annotate_chains: If True, add chain number annotation to frames
            
        Returns:
            List of saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []
        
        frames_np = frames.squeeze(0).cpu().numpy()  # (K, C, H, W)
        
        for i, frame in enumerate(frames_np):
            # Convert from (C, H, W) to (H, W, C) and scale to [0, 255]
            frame_img = (frame.transpose(1, 2, 0) * 255).astype(np.uint8)
            
            # Add chain annotation if requested
            if annotate_chains and chain_labels is not None and i < len(chain_labels):
                frame_img = self._annotate_frame_with_chain(frame_img, chain_labels[i])
            
            # Save frame
            path = os.path.join(output_dir, f"{prefix}_{i+1:02d}.png")
            Image.fromarray(frame_img).save(path)
            saved_paths.append(path)
            
        return saved_paths
    
    def _annotate_frame_with_chain(self, frame: np.ndarray, chain_num: int) -> np.ndarray:
        """
        Add chain number annotation to a frame with red rectangle background.
        
        Args:
            frame: Frame as numpy array (H, W, C) in uint8 [0, 255]
            chain_num: Chain number to annotate
            
        Returns:
            Annotated frame as numpy array
        """
        # Make a copy to avoid modifying the original
        annotated = frame.copy()
        
        # Text settings: small but bold
        text = f"C{chain_num}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        thickness = 1
        
        # Convert RGB to BGR for OpenCV
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        
        # Calculate text size to determine rectangle bounds
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Rectangle position (top-right corner with small padding)
        padding = 1
        frame_width = annotated_bgr.shape[1]
        rect_x2 = frame_width
        rect_x1 = rect_x2 - text_width - padding
        rect_y1 = 0
        rect_y2 = rect_y1 + text_height + baseline + padding
        
        # Draw filled red rectangle background
        cv2.rectangle(annotated_bgr, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 255), -1)  # Red in BGR, -1 fills
        
        # Draw bold white text on top
        text_x = rect_x1 + padding
        text_y = rect_y1 + text_height + padding - 1
        cv2.putText(annotated_bgr, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)  # White
        
        # Convert back to RGB
        annotated = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        
        return annotated
    
    def save_video(
        self,
        frames: torch.Tensor,
        output_path: str,
        fps: int = 5,
        include_condition: Optional[torch.Tensor] = None,
        chain_labels: Optional[List[int]] = None,
        annotate_chains: bool = False,
    ) -> str:
        """
        Save generated frames as a video.
        
        Args:
            frames: Generated frames tensor (1, K, C, H, W) where C is 3 (RGB only), in [0, 1] range
            output_path: Output video path
            fps: Frames per second
            include_condition: Optional condition frame (1, C, H, W) in [-1, 1] range
            chain_labels: Optional list of chain numbers for each frame
            annotate_chains: If True, add chain number annotation to frames
            
        Returns:
            Path to saved video
        """
        frames_list = []
        
        # Optionally prepend condition frame
        if include_condition is not None:
            cond_np = include_condition.squeeze(0).cpu().numpy()  # Remove batch dim: (C, H, W)
            # Only take RGB channels (first 3) if there's a depth channel
            if cond_np.shape[0] == 4:
                cond_np = cond_np[:3]
            # Convert from [-1, 1] to [0, 1] then to [0, 255]
            cond_np = (cond_np + 1) / 2  # [-1, 1] -> [0, 1]
            cond_np = np.clip(cond_np, 0, 1)
            cond_img = (cond_np.transpose(1, 2, 0) * 255).astype(np.uint8)
            # Annotate condition frame as "C0" (input frame)
            if annotate_chains:
                cond_img = self._annotate_frame_with_chain(cond_img, 0)
            frames_list.append(cond_img)
        
        # Add generated frames
        frames_np = frames.squeeze(0).cpu().numpy()
        for i, frame in enumerate(frames_np):
            frame_img = (frame.transpose(1, 2, 0) * 255).astype(np.uint8)
            # Add chain annotation if requested
            if annotate_chains and chain_labels is not None and i < len(chain_labels):
                frame_img = self._annotate_frame_with_chain(frame_img, chain_labels[i])
            frames_list.append(frame_img)
        
        # Save as video
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        imageio.mimwrite(output_path, frames_list, fps=fps)
        
        return output_path


def main():
    parser = argparse.ArgumentParser(
        description="CLOVER Visual Planner - Generate visual plans for robotic manipulation"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="original_image_1.jpg",
        help="Path to the input/condition image"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Put the drumstick on the plate",
        help="Natural language manipulation instruction"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/visual_planner.pt",
        help="Path to visual planner checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=8,
        help="Number of sub-goal frames to generate (K=8 in paper)"
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=20,
        help="DDIM sampling steps (20 in paper for efficiency)"
    )
    parser.add_argument(
        "--guidance_weight",
        type=float,
        default=4.0,
        help="Classifier-free guidance weight"
    )
    parser.add_argument(
        "--replan_threshold",
        type=float,
        default=1.0,
        help="Distance threshold for replanning (DR=1.0 in paper)"
    )
    parser.add_argument(
        "--max_replan_attempts",
        type=int,
        default=5,
        help="Maximum replanning attempts"
    )
    parser.add_argument(
        "--with_depth",
        action="store_true",
        default=True,  # Model was trained with depth, uses MiDaS for estimation
        help="Whether the model was trained with depth (uses MiDaS DPT_Large for depth estimation)"
    )
    parser.add_argument(
        "--num_chains",
        type=int,
        default=10,
        help="Max number of planning chains (total frames ≤ 8 * num_chains). Default 10 for complex tasks."
    )
    parser.add_argument(
        "--convergence_threshold",
        type=float,
        default=0.001,
        help="Early stopping threshold: stop when consecutive frames have cosine distance < this value"
    )
    parser.add_argument(
        "--no_auto_stop",
        action="store_true",
        help="Disable automatic stopping when task appears complete (always use all num_chains)"
    )
    parser.add_argument(
        "--video_fps",
        type=int,
        default=5,
        help="FPS for output video"
    )
    parser.add_argument(
        "--annotate_chains",
        action="store_true",
        help="Annotate each frame with its chain number (red text at top-left)"
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 60)
    print("CLOVER Visual Planner")
    print("=" * 60)
    print(f"Input image: {args.image}")
    print(f"Instruction: {args.instruction}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    # Check if input image exists
    if not os.path.exists(args.image):
        print(f"Error: Input image not found: {args.image}")
        sys.exit(1)
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Initialize visual planner
    planner = CLOVERVisualPlannerInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
        num_frames=args.num_frames,
        sampling_steps=args.sampling_steps,
        guidance_weight=args.guidance_weight,
        with_depth=args.with_depth,
        replan_threshold=args.replan_threshold,
    )
    
    # Load and preprocess input image
    print("\nLoading input image...")
    condition_image = planner.preprocess_image(args.image)
    print(f"  Image shape: {condition_image.shape}")
    
    # Generate visual plan
    print(f"\nGenerating visual plan for: '{args.instruction}'")
    if args.num_chains > 1:
        auto_stop = not args.no_auto_stop
        print(f"  Using chained planning with up to {args.num_chains} chains (≤{8 * args.num_chains} frames)")
        print(f"  Auto-stop on convergence: {auto_stop} (threshold: {args.convergence_threshold})")
        generated_frames, success, num_attempts, chain_labels = planner.generate_chained_visual_plan(
            condition_image=condition_image,
            text_instruction=args.instruction,
            num_chains=args.num_chains,
            max_replan_attempts=args.max_replan_attempts,
            convergence_threshold=args.convergence_threshold,
            auto_stop=auto_stop,
        )
    else:
        generated_frames, success, num_attempts = planner.generate_visual_plan(
            condition_image=condition_image,
            text_instruction=args.instruction,
            max_replan_attempts=args.max_replan_attempts,
        )
        # Single chain, all frames belong to chain 1
        chain_labels = [1] * generated_frames.shape[1]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save individual sub-goal frames
    print("\nSaving sub-goal frames...")
    frame_paths = planner.save_frames(
        frames=generated_frames,
        output_dir=args.output_dir,
        prefix="subgoal",
        chain_labels=chain_labels,
        annotate_chains=args.annotate_chains,
    )
    for path in frame_paths:
        print(f"  Saved: {path}")
    
    # Save video
    print("\nSaving video...")
    video_path = os.path.join(args.output_dir, "visual_plan.mp4")
    planner.save_video(
        frames=generated_frames,
        output_path=video_path,
        fps=args.video_fps,
        include_condition=condition_image,
        chain_labels=chain_labels,
        annotate_chains=args.annotate_chains,
    )
    print(f"  Saved: {video_path}")
    
    # Also save as GIF for easy viewing
    gif_path = os.path.join(args.output_dir, "visual_plan.gif")
    planner.save_video(
        frames=generated_frames,
        output_path=gif_path,
        fps=args.video_fps,
        include_condition=condition_image,
        chain_labels=chain_labels,
        annotate_chains=args.annotate_chains,
    )
    print(f"  Saved: {gif_path}")
    
    # Save condition image for reference
    cond_path = os.path.join(args.output_dir, "condition_image.png")
    cond_np = condition_image.squeeze(0).cpu().numpy()  # Remove batch dim: (C, H, W)
    # Only take RGB channels (first 3) if there's a depth channel
    if cond_np.shape[0] == 4:
        # Also save the depth channel for visualization
        depth_np = cond_np[3]  # (H, W)
        # Convert from [-1, 1] to [0, 1] then to [0, 255]
        depth_np = (depth_np + 1) / 2  # [-1, 1] -> [0, 1]
        depth_np = np.clip(depth_np, 0, 1)
        depth_img = (depth_np * 255).astype(np.uint8)
        depth_path = os.path.join(args.output_dir, "depth_estimation.png")
        # Save as colormap for better visualization
        depth_colored = cv2.applyColorMap(depth_img, cv2.COLORMAP_INFERNO)
        cv2.imwrite(depth_path, depth_colored)
        print(f"  Saved: {depth_path}")
        
        cond_np = cond_np[:3]
    # Convert from [-1, 1] to [0, 1] then to [0, 255]
    cond_np = (cond_np + 1) / 2  # [-1, 1] -> [0, 1]
    cond_np = np.clip(cond_np, 0, 1)
    cond_img = (cond_np.transpose(1, 2, 0) * 255).astype(np.uint8)
    Image.fromarray(cond_img).save(cond_path)
    print(f"  Saved: {cond_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Plan generation: {'SUCCESS' if success else 'FALLBACK (used last attempt)'}")
    print(f"Number of attempts: {num_attempts}")
    total_frames = generated_frames.shape[1]
    if args.num_chains > 1:
        # Calculate actual chains used based on frames
        # Chain 1 = 8 frames, subsequent chains = 7 frames each (first frame skipped)
        actual_chains = 1 + max(0, (total_frames - 8 + 6) // 7)  # Approximate
        print(f"Generated {total_frames} sub-goal frames (from chained planning)")
    else:
        print(f"Generated {total_frames} sub-goal frames")
    print(f"Outputs saved to: {args.output_dir}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
