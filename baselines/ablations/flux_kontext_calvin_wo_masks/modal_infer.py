"""
Parallel batch inference for the bft_i1 ablation.

Runs the trained FluxKontext editing pipeline (LoRA = last checkpoint,
checkpoint-epoch-20) over every sample in calvin_abcd_ier_fer_gt_test_data/,
using the i1 input set (original_image + instruction ONLY; NO masks at all), and writes edited_image.png into each sample dir locally.

Parallelism: a Modal Cls with up to 10 warm B200 containers (per-account GPU
cap). Each container loads the model ONCE (from the volume's cached weights) and
pulls work chunks via .map(). Idempotent: samples already having edited_image.png
are skipped, so re-running resumes.

Auth: per-process env vars (MODAL_TOKEN_ID/SECRET) — global profile untouched.
Run:  modal run modal_infer.py            # all 1087 samples
      modal run modal_infer.py --start 1 --end 101   # a sub-range
"""
import os
import modal

# ---- ablation-specific config ------------------------------------------------
APP_NAME = "flux-kontext-bft-i1-infer"
VOLUME_NAME = "flux-kontext-calvin-bft-i1-volume"
PASS_INITIAL_MASK = False  # i1 feeds NO masks
PASS_FINAL_MASK = False    # i1 feeds NO masks
CHECKPOINT_DIR = "/data/models/kontext-i2i/checkpoint-epoch-20"
TESTDATA_DIRNAME = "calvin_abcd_ier_fer_gt_test_data"
NUM_SAMPLES_DEFAULT_END = 1088   # samples are 1..1087
MAX_WORKERS = 10
CHUNK = 25
# -----------------------------------------------------------------------------

BASE_MODEL = "black-forest-labs/FLUX.1-Kontext-dev"
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_TESTDATA = os.path.join(LOCAL_DIR, TESTDATA_DIRNAME)

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME)
hf_secret = modal.Secret.from_name("huggingface-secret")

CUDA_TAG = "12.8.1-devel-ubuntu24.04"
image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    .uv_pip_install("torch==2.8.0", "torchvision")
    .uv_pip_install(
        "accelerate", "diffusers", "transformers", "peft", "pillow", "numpy",
        "huggingface-hub", "hf_transfer", "safetensors", "sentencepiece",
        "protobuf",
    )
    .env({"HF_HOME": "/data/datasets/.cache", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_file(
        os.path.join(LOCAL_DIR, "pipeline_flux_kontext_multi_imgs.py"),
        "/root/pipeline_flux_kontext_multi_imgs.py",
    )
    .add_local_dir(LOCAL_TESTDATA, "/root/testdata")
)


def build_enhanced_prompt(edit_instruction: str) -> str:
    """Identical to flux_kontext_eval_utils.build_enhanced_prompt (the prompt the
    model was trained/evaluated with — unchanged for the ablations)."""
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


@app.cls(
    image=image,
    gpu="B200",
    volumes={"/data": vol},
    secrets=[hf_secret],
    max_containers=MAX_WORKERS,
    timeout=3600,
    scaledown_window=300,
)
class InferenceWorker:
    @modal.enter()
    def load(self):
        import sys
        import torch

        sys.path.insert(0, "/root")
        from pipeline_flux_kontext_multi_imgs import FluxKontextPipeline

        os.environ.setdefault("HF_TOKEN", os.environ.get("HF_TOKEN", ""))
        self.torch = torch
        print("Loading base pipeline from cache...", flush=True)
        self.pipe = FluxKontextPipeline.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16
        ).to("cuda")
        print(f"Loading LoRA from {CHECKPOINT_DIR} ...", flush=True)
        self.pipe.load_lora_weights(CHECKPOINT_DIR)
        try:
            self.pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass
        print("Worker ready.", flush=True)

    @modal.method()
    def process(self, indices: list[int]) -> dict[int, bytes]:
        import io
        from PIL import Image

        torch = self.torch
        out: dict[int, bytes] = {}
        for idx in indices:
            sd = f"/root/testdata/{idx}"
            try:
                orig = Image.open(f"{sd}/original_image.jpg").convert("RGB")
                with open(f"{sd}/edit_instruction.txt") as f:
                    instr = f.read().strip()
            except Exception as e:
                print(f"  [skip {idx}] input load failed: {e}", flush=True)
                continue

            prompt = build_enhanced_prompt(instr)
            W, H = orig.size
            call_args = dict(
                original_image=orig,
                num_inference_steps=28,
                height=H, width=W,
                _auto_resize=False,
                max_area=H * W,
            )
            if PASS_INITIAL_MASK:
                call_args["initial_edit_region_mask"] = Image.open(
                    f"{sd}/initial_edit_region_mask.jpg"
                ).convert("RGB")
            if PASS_FINAL_MASK:
                call_args["final_edit_region_mask"] = Image.open(
                    f"{sd}/final_edit_region_mask.jpg"
                ).convert("RGB")

            with torch.no_grad():
                pe, ppe, _tids = self.pipe.encode_prompt(prompt, prompt_2=None)
                pe = pe.to("cuda")
                ppe = ppe.to("cuda")
                gen = torch.Generator(device="cuda").manual_seed(0)
                with torch.autocast("cuda"):
                    res = self.pipe(
                        **call_args,
                        prompt_embeds=pe,
                        pooled_prompt_embeds=ppe,
                        generator=gen,
                    )
            img = res.images[0]
            if img.size != (W, H):
                img = img.resize((W, H), Image.BILINEAR)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            out[idx] = buf.getvalue()
        return out


@app.local_entrypoint()
def main(start: int = 1, end: int = NUM_SAMPLES_DEFAULT_END, chunk: int = CHUNK):
    indices = [
        i for i in range(start, end)
        if os.path.isdir(os.path.join(LOCAL_TESTDATA, str(i)))
    ]
    pending = [
        i for i in indices
        if not os.path.exists(os.path.join(LOCAL_TESTDATA, str(i), "edited_image.png"))
    ]
    print(f"[{APP_NAME}] total in range={len(indices)}, already done={len(indices)-len(pending)}, "
          f"to process={len(pending)}", flush=True)
    if not pending:
        print("Nothing to do.")
        return

    chunks = [pending[i:i + chunk] for i in range(0, len(pending), chunk)]
    print(f"Dispatching {len(chunks)} chunks across up to {MAX_WORKERS} workers...", flush=True)

    worker = InferenceWorker()
    done = 0
    for res in worker.process.map(chunks):
        for idx, b in res.items():
            with open(os.path.join(LOCAL_TESTDATA, str(idx), "edited_image.png"), "wb") as f:
                f.write(b)
            done += 1
        print(f"  written {done}/{len(pending)}", flush=True)
    print(f"[{APP_NAME}] DONE: wrote {done} edited images.", flush=True)
