# Installation

SGE-Goal's training and inference scripts run on [Modal](https://modal.com): each script builds
its own container image (CUDA, PyTorch, diffusers, …) and runs on cloud GPUs, so your local machine
only needs a light Python environment that can launch them. The data pipeline and the diagnostic
evaluator can also run locally on a GPU machine, with the conda environments described below.

## 1. Clone and fetch third-party code

```bash
git clone https://github.com/Riddhi-Chatterjee/SGE-Goal.git
cd SGE-Goal
bash third_party/fetch_third_party.sh        # XMem, Segment Anything, SuSIE, CLOVER at pinned commits
```

`fetch_third_party.sh xmem sam` fetches only what the data pipeline and evaluator need. See
[`third_party/README.md`](../../third_party/README.md) for the pinned commits and patches.

## 2. Launcher environment (Modal)

```bash
python -m venv .venv && source .venv/bin/activate
pip install modal huggingface_hub
modal token new                               # one-time browser login
```

Instead of `modal token new` you can export `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`; every
script that sets Modal credentials reads them from the environment and otherwise uses your default
Modal profile.

### Modal secrets

The containers download models from the Hugging Face Hub through Modal secrets:

```bash
modal secret create huggingface-secret   HF_TOKEN=<your-hf-token>
modal secret create huggingface-secret-2 HF_TOKEN=<your-hf-token>   # used by the FLUX.2 klein training scripts
```

The token's account must have accepted the licences of the gated base models:
[FLUX.1 Kontext [dev]](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) and
[FLUX.2 klein](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B).

### GPUs

Training scripts default to NVIDIA B200 (`GPU_TYPE = "B200"` near the top of each
`modal_*_train.py`); inference scripts use smaller GPUs such as L40S, L4 or A10G. Change the
`GPU_TYPE` constant if your Modal workspace does not have access to the default type.

## 3. Local environments (optional)

| Component | Environment | Python |
|---|---|---|
| Data pipeline | `conda env create -f data_pipeline/environment.yml` → `lgie_dp` | 3.12 |
| Diagnostic evaluator | `conda env create -f evaluator/environment.yml` → `lgie_eval` | 3.12 |
| FLUX.1 Kontext LIBERO training (local runs) | `pip install -r sge_goal/requirements.txt` | 3.10+ |
| I2I-Bench comparison | `pip install -r evaluator/i2i_bench/requirements.txt` | 3.10+ |

The conda files pin exact versions (PyTorch 2.9, transformers 4.57). Both need a CUDA GPU.

## 4. Model weights

Foundation-model weights that are not downloaded automatically (SAM ViT-H, XMem, LoFTR) and the
paths the code expects are listed in [`third_party/README.md`](../../third_party/README.md#model-weights).

SGE-Goal's own LoRA adapters are on Hugging Face:

```bash
huggingface-cli download RiddhiCh/SGE-Goal --local-dir sge-goal-checkpoints
```

## 5. Credentials used by individual scripts

| Variable | Used by |
|---|---|
| `HF_TOKEN` | uploading processed datasets (`data_pipeline/`) |
| `OPENAI_API_KEY` | GPT Image 1.5 baseline (`baselines/gpt_image_1_5/inference.py`) |
| `GEMINI_API_KEYS` | Gemini 3 Pro Image baseline; one key, or several comma-separated keys used round-robin |
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | optional; otherwise the default Modal profile is used |
| `MODAL_BIN` | optional path to the `modal` executable for scripts that call it as a subprocess |

Never commit these values.
