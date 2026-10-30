"""
Local (no Modal) goal-image synthesis with FLUX.1 Kontext + LoRA.

Same inputs, prompt and sampler settings as ``modal_flux_kontext_inference.py``,
but runs on the local GPU and writes ``edited_image.jpg`` next to each test case
in place.

Usage:
    python local_flux_kontext_inference.py \
        --data-dir /abs/path/to/test_data \
        --checkpoint-dir /abs/path/to/step3_goal_image
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
from flux_kontext_eval_utils import build_enhanced_prompt  # noqa: E402

NUM_INFERENCE_STEPS = 10  # matches modal_flux_kontext_inference.py


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
    parser = argparse.ArgumentParser(description="Local goal-image synthesis.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--low-vram", action="store_true",
                        help="Encode prompts once, park the text encoders on the CPU and keep only the "
                             "transformer + VAE on the GPU (for a shared GPU with < ~36 GB free).")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SGE_GEN_SEED", "0")))
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip test cases that already have edited_image.jpg (resume).")
    args = parser.parse_args()

    device = args.device
    data_dir = os.path.abspath(args.data_dir)
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)

    lora_path = os.path.join(checkpoint_dir, "pytorch_lora_weights.safetensors")
    if not os.path.isfile(lora_path):
        raise FileNotFoundError(f"LoRA weights file not found: {lora_path}")

    print("Loading base Flux Kontext pipeline...")
    pipeline = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16
    )
    pipeline.load_lora_weights(checkpoint_dir)
    if args.low_vram:
        pipeline.to("cpu")
    else:
        pipeline = pipeline.to(device)
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
            if (args.skip_existing and os.path.isfile(os.path.join(case_dir, "edited_image.jpg"))) \
                    or not os.path.isfile(instr_path):
                continue
            with open(instr_path, "r", encoding="utf-8") as f:
                pending_prompts.append(build_enhanced_prompt(f.readline().strip()))
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
        out_path = os.path.join(case_dir, "edited_image.jpg")
        if args.skip_existing and os.path.isfile(out_path):
            skipped += 1
            continue

        orig_path = find_image(case_dir, "original_image")
        init_mask_path = find_image(case_dir, "initial_edit_region_mask")
        final_mask_path = find_image(case_dir, "final_edit_region_mask")
        instr_path = os.path.join(case_dir, "edit_instruction.txt")
        if None in (orig_path, init_mask_path, final_mask_path) or not os.path.isfile(instr_path):
            print(f"[{case_id}] missing inputs, skipping")
            failed += 1
            continue

        orig_image = Image.open(orig_path).convert("RGB")
        init_mask = Image.open(init_mask_path).convert("RGB")
        final_mask = Image.open(final_mask_path).convert("RGB")
        with open(instr_path, "r", encoding="utf-8") as f:
            edit_instruction = f.readline().strip()  # first line only; others may be metadata
        prompt = build_enhanced_prompt(edit_instruction)
        orig_w, orig_h = orig_image.size

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
                final_edit_region_mask=final_mask,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                num_inference_steps=NUM_INFERENCE_STEPS,
                height=orig_h,
                width=orig_w,
                _auto_resize=False,
                max_area=orig_h * orig_w,
                generator=generator,
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

        if pred_img.size != (orig_w, orig_h):
            pred_img = pred_img.resize((orig_w, orig_h), resample=Image.BILINEAR)
        pred_img.save(out_path, "JPEG", quality=95)
        done += 1
        print(f"[{case_id}] goal image done in {elapsed:.2f}s  ({done} done, {skipped} skipped, "
              f"{len(test_cases) - done - skipped - failed} left)", flush=True)

    print(f"Synthesis complete: {done} generated, {skipped} skipped, {failed} failed, "
          f"{time.perf_counter() - t_total:.0f}s total.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
