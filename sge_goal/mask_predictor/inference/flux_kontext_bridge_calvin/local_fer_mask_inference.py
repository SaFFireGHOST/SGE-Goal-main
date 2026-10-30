"""
Local (no Modal) Final Edit Region (FER) mask prediction with FLUX.1 Kontext + LoRA.

Same inputs, prompt, resolution handling and sampler settings as
``modal_fer_mask_inference.py``, but runs on the local GPU and writes
``final_edit_region_mask.jpg`` next to each test case in place.

Usage:
    python local_fer_mask_inference.py \
        --data-dir /abs/path/to/test_data \
        --checkpoint-dir /abs/path/to/step2_final_mask
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_flux_kontext_multi_imgs import FluxKontextPipeline  # noqa: E402

# Matches modal_fer_mask_inference.py.
INFERENCE_CONFIG = {
    "pretrained_model_name_or_path": "black-forest-labs/FLUX.1-Kontext-dev",
    "guidance_scale": 1.0,
    "num_inference_steps": 10,
    "resolution": 256,
    "seed": 0,
    "max_sequence_length": 512,
}

# The prompt template appended to every edit instruction during training.
DEFAULT_INSTRUCTION = """TASK: Edit Region Mask Prediction for Robotic Manipulation

You are provided with the following inputs:
1. ORIGINAL IMAGE: A scene containing objects, including a robotic arm and dynamic objects that can be manipulated.
2. INITIAL EDIT REGION MASK: A binary mask (white=1, black=0) showing the current locations of dynamic objects in the original image. White pixels indicate regions occupied by objects that will be manipulated/moved.
3. EDIT INSTRUCTION (above): A text description of the manipulation/edit to perform on the dynamic objects.

YOUR TASK: Generate a FINAL EDIT REGION MASK - a binary mask (white=1, black=0) predicting where the dynamic objects should appear AFTER the edit instruction is executed.
- White pixels (1): Regions where dynamic objects will be located after manipulation
- Black pixels (0): Background regions that remain unchanged

IMPORTANT:
- Focus ONLY on predicting object locations, not colors or textures
- The mask should be consistent across all RGB channels
- Use the initial mask to understand which objects are dynamic
- Use the edit instruction to determine the target locations
- Output should be a clean binary mask showing final object positions"""


def find_image(directory, base_name):
    for ext in ("jpg", "jpeg", "png"):
        path = os.path.join(directory, f"{base_name}.{ext}")
        if os.path.isfile(path):
            return path
    return None


def encode_prompts_once(pipeline, prompts, device):
    """Encode every distinct prompt once, then park the text encoders on the CPU.

    Prompts depend only on the instruction text (a few hundred distinct strings for
    1087 rows), so the 9.5 GB T5 encoder does not need to stay on a shared GPU.
    """
    pipeline.text_encoder.to(device)
    pipeline.text_encoder_2.to(device)
    cache = {}
    with torch.no_grad():
        for prompt in sorted(set(prompts)):
            embeds, pooled, _ = pipeline.encode_prompt(prompt, prompt_2=None)
            cache[prompt] = (embeds.cpu(), pooled.cpu())
    pipeline.text_encoder.to("cpu")
    pipeline.text_encoder_2.to("cpu")
    torch.cuda.empty_cache()
    return cache


def main():
    parser = argparse.ArgumentParser(description="Local FER mask prediction.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--low-vram", action="store_true",
                        help="Encode prompts once, park the text encoders on the CPU and keep only the "
                             "transformer + VAE on the GPU (for a shared GPU with < ~36 GB free).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip test cases that already have final_edit_region_mask.jpg (resume).")
    parser.add_argument("--seed", type=int, default=INFERENCE_CONFIG["seed"],
                        help="Base seed; case i uses seed + i. Change it to draw a different mask sample.")
    args = parser.parse_args()

    cfg = INFERENCE_CONFIG
    device = args.device
    data_dir = os.path.abspath(args.data_dir)
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)

    lora_path = os.path.join(checkpoint_dir, "pytorch_lora_weights.safetensors")
    if not os.path.isfile(lora_path):
        raise FileNotFoundError(f"LoRA weights file not found: {lora_path}")

    print(f"Loading base model: {cfg['pretrained_model_name_or_path']}")
    pipeline = FluxKontextPipeline.from_pretrained(
        cfg["pretrained_model_name_or_path"], torch_dtype=torch.bfloat16
    )
    pipeline.load_lora_weights(checkpoint_dir, weight_name="pytorch_lora_weights.safetensors")
    if args.low_vram:
        pipeline.to("cpu")
    else:
        pipeline = pipeline.to(device, dtype=torch.bfloat16)
    pipeline.set_progress_bar_config(disable=True)

    test_cases = sorted(
        (d for d in os.listdir(data_dir)
         if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith(".")),
        key=lambda x: int(x) if x.isdigit() else x,
    )
    print(f"Found {len(test_cases)} test cases.")

    prompt_cache = None
    if args.low_vram:
        pending_prompts = []
        for case_id in test_cases:
            case_dir = os.path.join(data_dir, case_id)
            instr_path = os.path.join(case_dir, "edit_instruction.txt")
            if (args.skip_existing and os.path.isfile(os.path.join(case_dir, "final_edit_region_mask.jpg"))) \
                    or not os.path.isfile(instr_path):
                continue
            with open(instr_path, "r", encoding="utf-8") as f:
                pending_prompts.append(f.read().strip() + "\n\n" + DEFAULT_INSTRUCTION.strip())
        print(f"Encoding {len(set(pending_prompts))} distinct prompts, then moving the transformer to the GPU...")
        prompt_cache = encode_prompts_once(pipeline, pending_prompts, device)
        pipeline.transformer.to(device)
        pipeline.vae.to(device)
        # The text encoders now live on the CPU, so pin the pipeline's execution device to the GPU.
        pipeline.__class__ = type(
            pipeline.__class__.__name__, (pipeline.__class__,),
            {"_execution_device": property(lambda self: torch.device(device))},
        )

    done = skipped = failed = 0
    t_total = time.perf_counter()
    for idx, case_id in enumerate(test_cases):
        case_dir = os.path.join(data_dir, case_id)
        out_path = os.path.join(case_dir, "final_edit_region_mask.jpg")
        if args.skip_existing and os.path.isfile(out_path):
            skipped += 1
            continue

        orig_path = find_image(case_dir, "original_image")
        mask_path = find_image(case_dir, "initial_edit_region_mask")
        instr_path = os.path.join(case_dir, "edit_instruction.txt")
        if orig_path is None or mask_path is None or not os.path.isfile(instr_path):
            print(f"[{case_id}] missing inputs, skipping")
            failed += 1
            continue

        orig_image = Image.open(orig_path).convert("RGB")
        init_mask = Image.open(mask_path).convert("RGB")
        with open(instr_path, "r", encoding="utf-8") as f:
            edit_instruction = f.read().strip()
        prompt = edit_instruction + "\n\n" + DEFAULT_INSTRUCTION.strip()

        # Target size: original size rounded down to a multiple of 32, area capped at resolution^2.
        multiple_of = 32
        target_width = (orig_image.size[0] // multiple_of) * multiple_of
        target_height = (orig_image.size[1] // multiple_of) * multiple_of
        max_area = target_width * target_height
        cap_area = cfg["resolution"] ** 2
        if max_area > cap_area:
            aspect = target_width / target_height
            target_width = int((cap_area * aspect) ** 0.5)
            target_height = int((cap_area / aspect) ** 0.5)
            target_width = (target_width // multiple_of) * multiple_of
            target_height = (target_height // multiple_of) * multiple_of
            max_area = target_width * target_height

        # Per-case seed (instead of one shared generator) so --skip-existing resumes reproducibly.
        generator = torch.Generator(device=device).manual_seed(args.seed + idx)

        t0 = time.perf_counter()
        with torch.no_grad():
            if prompt_cache is not None:
                prompt_embeds, pooled_prompt_embeds = prompt_cache[prompt]
            else:
                prompt_embeds, pooled_prompt_embeds, _ = pipeline.encode_prompt(prompt, prompt_2=None)
            prompt_embeds = prompt_embeds.to(device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(device)
        with torch.autocast("cuda"):
            out = pipeline(
                original_image=orig_image,
                initial_edit_region_mask=init_mask,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                generator=generator,
                height=target_height,
                width=target_width,
                guidance_scale=cfg["guidance_scale"],
                num_inference_steps=cfg["num_inference_steps"],
                max_sequence_length=cfg["max_sequence_length"],
                _auto_resize=False,
                max_area=max_area,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        if hasattr(out, "images"):
            pred_img = out.images[0]
        elif isinstance(out, (list, tuple)):
            pred_img = out[0][0] if isinstance(out[0], (list, tuple)) else out[0]
        else:
            pred_img = out
        if isinstance(pred_img, np.ndarray):
            pred_img = Image.fromarray(pred_img.astype(np.uint8)).convert("RGB")
        if pred_img is None:
            print(f"[{case_id}] pipeline returned None")
            failed += 1
            continue

        orig_w, orig_h = orig_image.size
        if pred_img.size != (orig_w, orig_h):
            pred_img = pred_img.resize((orig_w, orig_h), resample=Image.BILINEAR)
        pred_img.save(out_path, quality=95)
        done += 1
        print(f"[{case_id}] FER done in {elapsed:.2f}s  ({done} done, {skipped} skipped, "
              f"{len(test_cases) - done - skipped - failed} left)", flush=True)

    print(f"FER complete: {done} generated, {skipped} skipped, {failed} failed, "
          f"{time.perf_counter() - t_total:.0f}s total.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
