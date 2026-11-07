"""
Run sge_goal/final_mask/sample_step2.py on Modal (Step 2 samples scored by IoU, for IoU-ranked Flow-DPO pairs).

Uses the Variant A volume ``sge-variant-a`` (FLUX.1-Kontext-dev in /vol/hf, HF CALVIN tuples in /vol/calvin, see
sge_goal/variant_a/modal_train_flux.py). Extra uploads, once, from the repo root:
    modal volume put sge-variant-a run/ckpts/sge-goal-checkpoints/flux1-kontext-dev/calvin/step2_final_mask /released/step2_final_mask
    modal volume put sge-variant-a run/calvin/hf_dataset/pred_ier/train.parquet /pred_ier/

Usage:
    modal run sge_goal/final_mask/modal_sample_step2.py --name timing --args "--num-rows 64 --batch-rows 8"
GPU / timeout via env: FM_GPU (default H100), FM_TIMEOUT_H (default 1).
Fetch results: modal volume get sge-variant-a step2_samples/<name> run/final_mask/step2_samples/
"""

import os
import shlex
import subprocess

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
GPU = os.environ.get("FM_GPU", "H100")
TIMEOUT_H = float(os.environ.get("FM_TIMEOUT_H", "1"))
VOL = "/vol"
R = "/repo"

app = modal.App("sge-final-mask")
vol = modal.Volume.from_name("sge-variant-a")
secret = modal.Secret.from_name("huggingface-secret")  # HF_TOKEN (FLUX.1-Kontext-dev is gated; weights are cached)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", "torchvision==0.23.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("diffusers==0.36.0", "peft==0.18.1", "transformers==4.46.3", "accelerate==1.15.0", "safetensors",
                 "sentencepiece", "protobuf", "pyarrow", "pillow", "numpy")
    .env({"HF_HOME": f"{VOL}/hf", "TOKENIZERS_PARALLELISM": "false",
          "PYTHONWARNINGS": "ignore"})
    .add_local_dir(HERE, f"{R}/sge_goal/final_mask", ignore=["__pycache__"])
    .add_local_dir(os.path.join(REPO, "sge_goal/mask_predictor/inference/flux_kontext_bridge_calvin"),
                   f"{R}/sge_goal/mask_predictor/inference/flux_kontext_bridge_calvin", ignore=["__pycache__"])
)


# The pipeline is loaded into CPU RAM first (~34 GB bf16: transformer + T5), hence 48 GiB.
@app.function(image=image, gpu=GPU, volumes={VOL: vol}, secrets=[secret], timeout=int(TIMEOUT_H * 3600), cpu=4, memory=49152)
def sample(name: str, args: list):
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=False)
    cmd = ["python", "-u", f"{R}/sge_goal/final_mask/sample_step2.py",
           "--out-dir", f"{VOL}/step2_samples/{name}",
           "--train-parquet", f"{VOL}/calvin/data/train-*.parquet",
           "--pred-ier", f"{VOL}/pred_ier/train.parquet",
           "--checkpoint-dir", f"{VOL}/released/step2_final_mask", *args]
    print("Running:", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, check=True)
    finally:
        vol.commit()


@app.local_entrypoint()
def main(name: str, args: str = ""):
    sample.remote(name, shlex.split(args))
