"""
Run sge_goal/variant_a/flux_lora_trainer.py on Modal (A-dense; also usable for sup / dpo).

Volume ``sge-variant-a`` (mounted at /vol) holds everything; uploads, once, from the repo root:
    modal volume put sge-variant-a run/eval_inputs/ckpts/gr_mg/epoch=47-step=83712.ckpt /gr_mg/
    modal volume put sge-variant-a run/ckpts/sge-goal-checkpoints/flux1-kontext-dev/calvin/step3_goal_image /released/step3_goal_image
    modal volume put sge-variant-a run/calvin/hf_dataset/pred_ier/validation.parquet /pred_ier/
    modal volume put sge-variant-a run/calvin/d_train /d_train
    modal volume put sge-variant-a <CALVIN-EVALS folder with M_i / M_f per demo> /beh/<name>/calvin_abcd_test_data
FLUX.1-Kontext-dev (gated) and the HF CALVIN tuples (RiddhiCh/Calvin-Dataset-Final) are downloaded into
the volume by a CPU-only ``prepare`` call (secret ``huggingface-secret`` with HF_TOKEN), so GPU time is
not spent on downloads.

Usage:
    modal run sge_goal/variant_a/modal_train_flux.py::prepare
    modal run sge_goal/variant_a/modal_train_flux.py --args "--step 3 --mode dense --name t --beh-dir /vol/beh/pilot_m0a0/calvin_abcd_test_data --max-steps 6"
GPU / timeout via env: VA_GPU (default H100), VA_TIMEOUT_H (default 4).
Fetch results: modal volume get sge-variant-a training/<name> run/variant_a/training/
"""

import os
import shlex
import subprocess

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
GPU = os.environ.get("VA_GPU", "H100")
TIMEOUT_H = float(os.environ.get("VA_TIMEOUT_H", "4"))
VOL = "/vol"
R = "/repo"  # repo layout inside the container, so the trainer's relative paths resolve

app = modal.App("sge-variant-a")
vol = modal.Volume.from_name("sge-variant-a", create_if_missing=True)
secret = modal.Secret.from_name("huggingface-secret")  # HF_TOKEN

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", "torchvision==0.23.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("diffusers==0.36.0", "peft==0.18.1", "transformers==4.46.3", "accelerate==1.15.0", "safetensors",
                 "sentencepiece", "protobuf", "pyarrow", "pillow", "numpy", "scikit-image", "tqdm",
                 "huggingface_hub", "hf_transfer", "einops", "ftfy", "regex")
    .run_commands("pip install --no-deps flamingo-pytorch==0.1.2 einops-exts==0.0.4 openai-clip==1.0.1")
    .env({"HF_HOME": f"{VOL}/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTHONWARNINGS": "ignore::FutureWarning"})
    .add_local_dir(HERE, f"{R}/sge_goal/variant_a", ignore=["__pycache__"])
    .add_local_dir(os.path.join(REPO, "sge_goal/synthesis/inference/flux_kontext_bridge_calvin"),
                   f"{R}/sge_goal/synthesis/inference/flux_kontext_bridge_calvin", ignore=["__pycache__"])
    .add_local_dir(os.path.join(REPO, "sge_goal/mask_predictor/inference/flux_kontext_bridge_calvin"),
                   f"{R}/sge_goal/mask_predictor/inference/flux_kontext_bridge_calvin", ignore=["__pycache__"])
    .add_local_dir(os.path.join(REPO, "downstream/calvin/gr_mg/policy"), f"{R}/downstream/calvin/gr_mg/policy",
                   ignore=["__pycache__"])
    .add_local_dir(os.path.join(REPO, "downstream/calvin/gr_mg/evaluate/utils"),
                   f"{R}/downstream/calvin/gr_mg/evaluate/utils", ignore=["__pycache__"])
)


@app.function(image=image, volumes={VOL: vol}, secrets=[secret], timeout=3600, cpu=4, memory=16384)
def prepare():
    """CPU only: FLUX.1-Kontext-dev weights + HF CALVIN tuples into the volume (skips what is there)."""
    from huggingface_hub import snapshot_download

    snapshot_download("black-forest-labs/FLUX.1-Kontext-dev",
                      allow_patterns=["*.json", "*.txt", "transformer/*", "vae/*", "text_encoder/*",
                                      "text_encoder_2/*", "tokenizer/*", "tokenizer_2/*", "scheduler/*"])
    snapshot_download("RiddhiCh/Calvin-Dataset-Final", repo_type="dataset",
                      allow_patterns=["data/train-*.parquet", "data/validation-*.parquet"], local_dir=f"{VOL}/calvin")
    import clip  # ViT-B/32 for GR-MG's language encoder
    clip.load("ViT-B/32", device="cpu", download_root=f"{VOL}/clip")
    vol.commit()
    subprocess.run(["du", "-sh", f"{VOL}/hf", f"{VOL}/calvin", f"{VOL}/clip"], check=False)


# CPU / memory are billed too: 4 cores + 48 GiB fit A-dense (~9 GB of demo frames in RAM, 2 loader workers).
@app.function(image=image, gpu=GPU, volumes={VOL: vol}, secrets=[secret], timeout=int(TIMEOUT_H * 3600),
              cpu=4, memory=49152)
def train(args: list):
    os.makedirs(os.path.expanduser("~/.cache"), exist_ok=True)
    if not os.path.exists(os.path.expanduser("~/.cache/clip")):
        os.symlink(f"{VOL}/clip", os.path.expanduser("~/.cache/clip"))  # clip.load's default download_root
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=False)
    cmd = ["python", "-u", f"{R}/sge_goal/variant_a/flux_lora_trainer.py",
           "--out-dir", f"{VOL}/training", "--num-workers", "2",
           "--train-parquet", f"{VOL}/calvin/data/train-*.parquet",
           "--val-parquet", f"{VOL}/calvin/data/validation-*.parquet",
           "--pred-ier-dir", f"{VOL}/pred_ier", "--d-train-dir", f"{VOL}/d_train",
           "--grmg-ckpt", f"{VOL}/gr_mg/epoch=47-step=83712.ckpt",
           "--init-lora", f"{VOL}/released/step3_goal_image", *args]
    print("Running:", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, check=True)
    finally:
        vol.commit()


@app.local_entrypoint()
def main(args: str = ""):
    train.remote(shlex.split(args))
