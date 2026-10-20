"""
Parallel Modal inference for the Calvin BFT ablations.

Runs the trained Flux 2 Klein LoRA (last checkpoint, epoch 20) over a shard of the
test samples in ./calvin_abcd_ier_fer_gt_test_data, writing edited_image.jpg into
each sample's directory.

The ablation is auto-detected from this file's parent folder name:
  * ...bft_i2  -> conditioning = [original, initial_edit_region_mask]   (final mask dropped)
  * ...bft_i1  -> conditioning = [original]                             (both masks dropped)

Design (fast + correct):
  * Lean image (no flash-attn; SDPA is used) -> quick build.
  * LoRA weights baked into the image; base model cached on a prewarmed HF-cache Volume
    so the 10 GPU workers don't each re-download ~35 GB.
  * A modal.Cls loads the pipeline ONCE per container (@enter); .map() reuses up to 10
    GPU containers and amortizes model load over the shard.
  * Inputs are read locally and outputs returned as bytes (written locally) -> no output
    volume / download / merge. Idempotent: samples that already have edited_image.jpg are
    skipped, so re-running fills only the gaps.

Auth uses MODAL_TOKEN_ID / MODAL_TOKEN_SECRET from the environment (never the global
profile). Launch via run_inference.sh which sets the right account credentials + shard.

Usage (one account, one shard):
    modal run modal_infer.py --shard-start 1 --shard-end 362
"""

import os
import modal

# ----------------------------------------------------------------------------
# Auto-detect which ablation this folder is.
# ----------------------------------------------------------------------------
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
_folder = os.path.basename(LOCAL_DIR.rstrip("/"))
ABLATION = "i2" if _folder.endswith("i2") else "i1"
USE_INIT_MASK = ABLATION == "i2"          # i2 keeps the initial mask; i1 keeps nothing
MODEL_DIR_NAME = f"flux-2-klein-lora-calvin-{ABLATION}"
LAST_CHECKPOINT = "checkpoint-epoch-20"   # max checkpoint for both runs

BASE_MODEL = "black-forest-labs/FLUX.2-klein-base-9B"
TEST_DATA_LOCAL = os.path.join(LOCAL_DIR, "calvin_abcd_ier_fer_gt_test_data")
LORA_LOCAL = os.path.join(
    LOCAL_DIR, "models", MODEL_DIR_NAME, LAST_CHECKPOINT, "pytorch_lora_weights.safetensors"
)

# Inference recipe — matches the validated run_validation_or_test path of training.
NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 1.0
GPU_TYPE = "B200"
MAX_CONTAINERS = 10  # Modal per-account GPU concurrency limit

APP_NAME = f"flux2-klein-infer-{ABLATION}"
HF_CACHE_VOLUME = "flux2-klein-infer-hf-cache"

# ----------------------------------------------------------------------------
# Image (lean: no flash-attn). LoRA baked in at /root/lora.
# ----------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install("torch==2.8.0", "torchvision")
    .uv_pip_install(
        "transformers",
        "peft",
        "accelerate",
        "safetensors",
        "huggingface-hub",
        "hf_transfer",
        "pillow",
        "numpy",
        "sentencepiece",
        "protobuf",
    )
    .run_commands("pip install git+https://github.com/huggingface/diffusers.git")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/cache",
        }
    )
    .add_local_file(LORA_LOCAL, remote_path="/root/lora/pytorch_lora_weights.safetensors")
)

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")  # provides HF_TOKEN


def _build_enhanced_prompt(edit_instruction: str) -> str:
    """Identical to the prompt builder used during training/validation."""
    return f"""{edit_instruction}

Task: Robotic Manipulation Image Editing

You are performing an image editing task in the domain of robotic manipulation. Your goal is to generate the final edited scene after a robotic arm performs the manipulation action described above.

Inputs Provided:
1. Original Image: The initial scene before any manipulation occurs. This is your reference for the scene layout, lighting, and static elements.
2. Initial Edit Region Mask: A binary mask highlighting the dynamic object(s) in the original image that will be manipulated by the robotic arm. White pixels (value 1) indicate dynamic objects; black pixels (value 0) indicate static background.
3. Final Edit Region Mask: A binary mask showing where the dynamic object(s) should appear in the final edited image after manipulation. This serves as spatial guidance for object placement.

Expected Output:
Generate the final edited scene image where:
- The dynamic object(s) identified in the Initial Edit Region Mask have been moved/manipulated according to the edit instruction
- The object(s) must appear precisely in the regions indicated by the Final Edit Region Mask
- All static background elements from the original image should be preserved exactly
- The edited image should look photorealistic and physically plausible
- Maintain consistent lighting, shadows, and visual style with the original image"""


def _resolve_pipeline_cls():
    import importlib
    import diffusers

    cls = getattr(diffusers, "Flux2KleinPipeline", None)
    if cls is None:
        for mod in ("diffusers.pipelines", "diffusers.pipelines.flux_2_klein"):
            try:
                cls = getattr(importlib.import_module(mod), "Flux2KleinPipeline", None)
            except Exception:
                cls = None
            if cls is not None:
                break
    if cls is None:
        raise ImportError("Flux2KleinPipeline not found in installed diffusers build.")
    return cls


# ----------------------------------------------------------------------------
# Prewarm: download the base model into the HF-cache volume once per account so
# the GPU workers load from cache instead of each re-downloading ~35 GB.
# ----------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={"/cache": hf_cache},
    secrets=[hf_secret],
    timeout=3600,
    memory=32768,
)
def prewarm():
    import os
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"Prewarming HF cache for {BASE_MODEL} (token={'present' if token else 'MISSING'})...")
    snapshot_download(BASE_MODEL, token=token)
    hf_cache.commit()
    print("Prewarm complete; HF cache committed.")
    return True


# ----------------------------------------------------------------------------
# Inferencer: loads the pipeline once per container, processes many samples.
# ----------------------------------------------------------------------------
@app.cls(
    image=image,
    gpu=f"{GPU_TYPE}:1",
    volumes={"/cache": hf_cache},
    secrets=[hf_secret],
    timeout=86400,
    memory=65536,
    max_containers=MAX_CONTAINERS,
    retries=2,
)
class Inferencer:
    @modal.enter()
    def load(self):
        import torch

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        Flux2KleinPipeline = _resolve_pipeline_cls()
        print(f"[{ABLATION}] Loading {BASE_MODEL} ...")
        self.pipe = Flux2KleinPipeline.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, token=token
        )
        print(f"[{ABLATION}] Loading LoRA weights from /root/lora ({LAST_CHECKPOINT}) ...")
        self.pipe.load_lora_weights("/root/lora")
        self.pipe.to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        self.torch = torch
        print(f"[{ABLATION}] Pipeline ready. USE_INIT_MASK={USE_INIT_MASK}")

    @modal.method()
    def infer(self, payload: dict) -> dict:
        import io
        from PIL import Image

        torch = self.torch
        sid = int(payload["id"])
        original = Image.open(io.BytesIO(payload["original"])).convert("RGB")
        cond_images = [original]
        if USE_INIT_MASK:
            init_mask = Image.open(io.BytesIO(payload["init_mask"])).convert("RGB")
            cond_images = [original, init_mask]

        prompt = _build_enhanced_prompt(payload["instruction"])
        generator = torch.Generator(device="cuda").manual_seed(sid)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            image_out = self.pipe(
                prompt=prompt,
                image=cond_images,
                height=original.height,
                width=original.width,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=generator,
            ).images[0]

        buf = io.BytesIO()
        image_out.save(buf, format="JPEG", quality=95)
        return {"id": sid, "edited": buf.getvalue()}


# ----------------------------------------------------------------------------
# Local entrypoint: read this account's shard locally, run the map, write results.
# ----------------------------------------------------------------------------
@app.local_entrypoint()
def main(shard_start: int = 1, shard_end: int = 1087):
    print("=" * 70)
    print(f"Ablation {ABLATION} | shard [{shard_start}, {shard_end}] | folder {LOCAL_DIR}")
    print(f"Conditioning: {'[original, initial_mask]' if USE_INIT_MASK else '[original]'}")
    print("=" * 70)

    payloads = []
    skipped_done = 0
    missing = 0
    for sid in range(shard_start, shard_end + 1):
        d = os.path.join(TEST_DATA_LOCAL, str(sid))
        if not os.path.isdir(d):
            continue
        out_path = os.path.join(d, "edited_image.jpg")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            skipped_done += 1
            continue
        try:
            with open(os.path.join(d, "original_image.jpg"), "rb") as f:
                original = f.read()
            with open(os.path.join(d, "edit_instruction.txt")) as f:
                instruction = f.read().strip()
        except FileNotFoundError:
            missing += 1
            continue
        p = {"id": sid, "original": original, "instruction": instruction}
        if USE_INIT_MASK:
            with open(os.path.join(d, "initial_edit_region_mask.jpg"), "rb") as f:
                p["init_mask"] = f.read()
        payloads.append(p)

    print(f"To process: {len(payloads)} | already done: {skipped_done} | missing inputs: {missing}")
    if not payloads:
        print("Nothing to do for this shard.")
        return

    print("Prewarming model cache (one-time per account)...")
    prewarm.remote()

    print(f"Running inference on {len(payloads)} samples across up to {MAX_CONTAINERS} GPUs...")
    written = 0
    for res in Inferencer().infer.map(payloads, order_outputs=False):
        sid = res["id"]
        out_path = os.path.join(TEST_DATA_LOCAL, str(sid), "edited_image.jpg")
        with open(out_path, "wb") as f:
            f.write(res["edited"])
        written += 1
        if written % 25 == 0:
            print(f"  wrote {written}/{len(payloads)} (latest id={sid})")

    print(f"DONE: shard [{shard_start}, {shard_end}] wrote {written} edited images.")
