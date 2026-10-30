"""
Train the learned IER mask predictor on Modal with the same code as train_ier_mask.py.

Data (RiddhiCh/Calvin-Dataset-Final) and EVF-SAM weights are downloaded inside the
container into the ``sge-learned-ier`` volume (cached across runs); checkpoints and
logs are written to ``ier_training/<name>/`` in that volume.

Usage (from the repo root):
    modal run sge_goal/grounding/learned_ier/modal_train_ier.py \
        --args "--name evf_arm_object --heads arm_object --epochs 3 --batch-size 8"

GPU and timeout via env: IER_GPU (default A100-40GB), IER_TIMEOUT_H (default 12).

Fetch results:
    modal volume get sge-learned-ier ier_training/evf_arm_object run/ier_training/
"""

import os
import shlex
import subprocess

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
EVF_LOCAL = os.path.join(REPO, "run", "third_party", "EVF-SAM")

GPU = os.environ.get("IER_GPU", "A100-40GB")
TIMEOUT_H = float(os.environ.get("IER_TIMEOUT_H", "12"))
VOL = "/vol"

app = modal.App("sge-learned-ier")
vol = modal.Volume.from_name("sge-learned-ier", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", "torchvision==0.23.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers==4.46.3", "accelerate==1.15.0", "timm==1.0.30", "sentencepiece", "protobuf", "pyarrow",
                 "scipy", "einops", "pillow", "numpy", "huggingface_hub")
    .run_commands("pip install --no-deps torchscale==0.2.0 fairscale==0.4.13")
    .env({"HF_HOME": f"{VOL}/hf", "PYTHONPATH": "/root/evf", "TOKENIZERS_PARALLELISM": "false",
          "PYTHONWARNINGS": "ignore::FutureWarning"})
    .add_local_dir(EVF_LOCAL, "/root/evf", ignore=[".git", "assets"])
    .add_local_dir(HERE, "/root/learned_ier", ignore=["__pycache__"])
)


@app.function(image=image, gpu=GPU, volumes={VOL: vol}, timeout=int(TIMEOUT_H * 3600), cpu=4, memory=24576)
def train(args: list):
    from huggingface_hub import snapshot_download

    snapshot_download("RiddhiCh/Calvin-Dataset-Final", repo_type="dataset",
                      allow_patterns=["data/*.parquet"], local_dir=f"{VOL}/calvin")
    vol.commit()
    data = f"{VOL}/calvin/data"
    cmd = ["python", "/root/learned_ier/train_ier_mask.py", "--evf-root", "/root/evf",
           "--out-dir", f"{VOL}/ier_training", "--num-workers", "4",
           "--train-parquet", f"{data}/train-*.parquet",
           "--val-parquet", f"{data}/validation-*.parquet",
           "--test-parquet", f"{data}/test-*.parquet", *args]
    print("Running:", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, check=True)
    finally:
        vol.commit()


@app.local_entrypoint()
def main(args: str = ""):
    train.remote(shlex.split(args))
